#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include "kiss.h"
#include "cblas.h"
#include "lapacke.h"
#include "mmio_dense.h"
#include "svd_routines.h"

typedef struct { int m, n, p, b, mode, seed; double cond, dmax; } params_t;

int iseed[4];
int iseed_init();
int log2i(int v);

int svdgen_param_check(int m, int n, int mode, double cond, double dmax);
int svdalg_param_check(int m, int n, int p, int b);
int params_init(params_t *ps);
int parse_params(int argc, char *argv[], params_t *ps);
int usage(char *argv[], const params_t *ps);
int gen_test_mat(double *A, double *S, int m, int n, int mode, double cond, double dmax);
int gen_uv_mats(const double *A, double *S, double *U, double *Vt, int m, int n);
double l2dist(const double *x, const double *y, int n);
int compute_errors(const double *A, const double *U, const double *Up, const double *S, const double *Sp, const double *Vt, const double *Vtp, int m, int n, int p, double errs[4]);
int svd_serial(double const *A, double *Up, double *Sp, double *Vtp, int m, int n, int p, int b);
double get_clock_telapsed(struct timespec start, struct timespec end);

int main(int argc, char *argv[])
{
    params_t params;
    struct timespec start, end;
    double gentime, svdtime, errtime;

    if (parse_params(argc, argv, &params) != 0)
        return 1;

    if (params.seed <= 0)
        kiss_init();
    else
        kiss_seed((uint32_t)params.seed);

    iseed_init();

    int m=params.m, n=params.n, p=params.p, b=params.b, mode=params.mode;
    double cond=params.cond, dmax=params.dmax;

    if (svdgen_param_check(m, n, mode, cond, dmax))
        return 1;

    if (svdalg_param_check(m, n, p, b))
        return 1;

    clock_gettime(CLOCK_MONOTONIC, &start);

    double *A = malloc(m*n*sizeof(double));
    double *S = malloc(n*sizeof(double));

    gen_test_mat(A, S, m, n, mode, cond, dmax);

    double *Scheck = malloc(n*sizeof(double));
    double *U = malloc(m*n*sizeof(double));
    double *Vt = malloc(n*n*sizeof(double));

    gen_uv_mats(A, Scheck, U, Vt, m, n);

    clock_gettime(CLOCK_MONOTONIC, &end);
    gentime = get_clock_telapsed(start, end);

    int show = n < 5? n : 5;
    fprintf(stderr, "[main:gen_truth::*] diag(S)[0..%d] = ", show-1);
    for (int i = 0; i < show; ++i) fprintf(stderr, "%.3e,", S[i]);
    fprintf(stderr, "%s\n", n < 5? "" : "...");

    fprintf(stderr, "[main:gen_truth::%.5f(s)] err=%.18e (DLATMS-vs-DGESVD) [S :: singular values]\n", gentime, l2dist(S, Scheck, n));
    free(Scheck);

    double *Up = malloc(m*p*sizeof(double));
    double *Sp = malloc(p*sizeof(double));
    double *Vtp = malloc(p*n*sizeof(double));

    clock_gettime(CLOCK_MONOTONIC, &start);

    svd_serial(A, Up, Sp, Vtp, m, n, p, b);

    clock_gettime(CLOCK_MONOTONIC, &end);
    svdtime = get_clock_telapsed(start, end);

    fprintf(stderr, "[main:svd_serial::%.5f(s)]\n", svdtime);

    double errs[4];

    clock_gettime(CLOCK_MONOTONIC, &start);

    compute_errors(A, U, Up, S, Sp, Vt, Vtp, m, n, p, errs);

    clock_gettime(CLOCK_MONOTONIC, &end);
    errtime = get_clock_telapsed(start, end);

    fprintf(stderr, "[main:compute_errors::%.5f(s)]\n", errtime);
    fprintf(stderr, "[main:compute_errors::*] err=%.18e (DLATMS-vs-RANDSVD) [A :: test matrix]\n", errs[0]);
    fprintf(stderr, "[main:compute_errors::*] err=%.18e (DLATMS-vs-RANDSVD) [S :: singular values]\n", errs[1]);
    fprintf(stderr, "[main:compute_errors::*] err=%.18e (DLATMS-vs-RANDSVD) [U :: singular vectors (left)]\n", errs[2]);
    fprintf(stderr, "[main:compute_errors::*] err=%.18e (DLATMS-vs-RANDSVD) [V :: singular vectors (right)]\n", errs[3]);

    free(A);
    free(S);
    free(Sp);
    free(U);
    free(Up);
    free(Vt);
    free(Vtp);

    return 0;
}

int compute_errors(const double *A,
                   const double *U, const double *Up,
                   const double *S, const double *Sp,
                   const double *Vt, const double *Vtp,
                   int m, int n, int p,
                   double errs[4])
{
    double Aerr, Serr, Uerr, Verr;
    double *mem = malloc(m*m*sizeof(double));

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
        {
            double acc = 0;

            for (int k = 0; k < p; ++k)
            {
                acc += Up[i + k*m]*Sp[k]*Vtp[k + j*p];
            }

            mem[i + j*m] = acc - A[i + j*m];
        }

    Aerr = LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', m, n, mem, m);

    Serr = l2dist(S, Sp, p);

    /*M = U@U.T - Up@Up.T m-by-m */

    for (int j = 0; j < m; ++j)
        for (int i = 0; i < m; ++i)
        {
            double acc = 0;

            for (int k = 0; k < p; ++k)
            {
                acc += U[i + k*m]*U[j + k*m];
                acc -= Up[i + k*m]*Up[j + k*m];
            }

            mem[i + j*m] = acc;
        }

    Uerr = LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', m, m, mem, m);

    /*M = Vt.T@Vt - Vtp.T@Vtp n-by-n */

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
        {
            double acc = 0;

            for (int k = 0; k < p; ++k)
            {
                acc += Vt[k + i*n]*Vt[k + j*n];
                acc -= Vtp[k + i*p]*Vtp[k + j*p];
            }

            mem[i + j*n] = acc;
        }

    Verr = LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', n, n, mem, n);

    free(mem);

    errs[0] = Aerr, errs[1] = Serr, errs[2] = Uerr, errs[3] = Verr;

    return 0;
}

int log2i(int v)
{
    int x = 0;
    while (v >>= 1) ++x;
    return x;
}

int iseed_init()
{
    for (int i = 0; i < 4; ++i) iseed[i] = (kiss_rand() % 4096);
    iseed[3] &= (iseed[3]^1); /* iseed[3] must be odd */
    return 0;
}

double l2dist(const double *x, const double *y, int n)
{
    double v = 0;

    for (int i = 0; i < n; ++i)
    {
        v += ((x[i] - y[i]) * (x[i] - y[i]));
    }

    return sqrt(v);
}

int params_init(params_t *ps)
{
    ps->m = 256;
    ps->n = 128;
    ps->p = 10;
    ps->b = 8;
    ps->mode = -1;
    ps->cond = 100.f;
    ps->dmax = 2.f;
    ps->seed = -1;

    return 0;
}

int usage(char *argv[], const params_t *ps)
{
    fprintf(stderr, "Usage: %s [options]\n", argv[0]);
    fprintf(stderr, "Options: -m INT    rows of test matrix [%d]\n", ps->m);
    fprintf(stderr, "         -n INT    columns of test matrix [%d]\n", ps->n);
    fprintf(stderr, "         -p INT    SVD truncation parameter [%d]\n", ps->p);
    fprintf(stderr, "         -b INT    number of 'compute nodes'[%d]\n", ps->b);
    fprintf(stderr, "         -u INT    matrix generation mode [%d]\n", ps->mode);
    fprintf(stderr, "         -c FLOAT  DLATMS cond or condition number [%.3e]\n", ps->cond);
    fprintf(stderr, "         -d FLOAT  DLATMS dmax or damping factor [%.3e]\n", ps->dmax);
    fprintf(stderr, "         -s INT    RNG seed [%d]\n", ps->seed);
    fprintf(stderr, "         -S FILE   output singular values file\n");
    fprintf(stderr, "         -h        help message\n");
    return 1;
}

int parse_params(int argc, char *argv[], params_t *ps)
{
    int c;
    params_init(ps);

    while ((c = getopt(argc, argv, "m:n:p:b:u:c:d:s:h")) >= 0)
    {
        if      (c == 'm') ps->m = atoi(optarg);
        else if (c == 'n') ps->n = atoi(optarg);
        else if (c == 'p') ps->p = atoi(optarg);
        else if (c == 'b') ps->b = atoi(optarg);
        else if (c == 'u') ps->mode = atoi(optarg);
        else if (c == 'c') ps->cond = atof(optarg);
        else if (c == 'd') ps->dmax = atof(optarg);
        else if (c == 's') ps->seed = atoi(optarg);
        else if (c == 'h')
        {
            params_init(ps);
            return usage(argv, ps);
        }
    }

    return 0;
}

int svdgen_param_check(int m, int n, int mode, double cond, double dmax)
{
    if (!(1 <= n && n <= m) || (m&(m-1)) || (n&(n-1)))
    {
        fprintf(stderr, "[error::svdgen_param_check][m=%d,n=%d] must have 1 <= n <= m with n and m both being powers of 2\n", m, n);
        return 1;
    }

    if (mode == 0 || mode > 5)
    {
        fprintf(stderr, "[error::svdgen_param_check][mode=%d] mode must be an integer between 1 and 5 inclusive or it must be negative\n", mode);
        return 1;
    }

    if (mode != 0 && (cond < 1 || dmax <= 0))
    {
        fprintf(stderr, "[error::svdgen_param_check][mode=%d,cond=%.5e,dmax=%.5e] must have cond >= 1 and dmax > 0 when mode=1,2,..,5\n", mode, cond, dmax);
        return 1;
    }

    if (mode < 0 && (cond <= 0 || dmax < 1))
    {
        fprintf(stderr, "[error::svdgen_param_check][mode=%d,cond=%.5e,damp=%.5e] must have cond > 0 and damp >= 1 when mode < 0\n", mode, cond, dmax);
        return 1;
    }

    return 0;
}

int svdalg_param_check(int m, int n, int p, int b)
{
    if (!(1 <= p && p <= n && n <= m) || (m&(m-1)) || (n&(n-1)))
    {
        fprintf(stderr, "[error::svdalg_param_check][m=%d,n=%d,p=%d] must have 1 <= p <= n <= m with n and m being powers of 2\n", m, n, p);
        return 1;
    }

    if (b <= 1 || (b&(b-1)) || n % b != 0)
    {
        fprintf(stderr, "[error:svdalg_param_check][b=%d,n=%d] must have b > 1 and n %% b == 0 with b being a power of 2\n", b, n);
        return 1;
    }

    return 0;
}

int gen_test_mat(double *A, double *S, int m, int n, int mode, double cond, double dmax)
{
    if (mode < 0)
    {
        S[0] = cond;

        for (int i = 1; i < n; ++i)
            S[i] = S[i-1] / dmax;

        mode = 0;
    }
    else
    {
        assert(1 <= mode && mode <= 5);
    }

    LAPACKE_dlatms(LAPACK_COL_MAJOR, m, n, 'U', iseed, 'N', S, mode, cond, dmax, m, n, 'N', A, m);

    return 0;
}

int gen_uv_mats(const double *A, double *S, double *U, double *Vt, int m, int n)
{
    double *Al = malloc(m*n*sizeof(double));
    double *work = malloc(5*n*sizeof(double));

    memcpy(Al, A, m*n*sizeof(double));

    LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'S', m, n, Al, m, S, U, m, Vt, n, work);

    free(Al);
    free(work);

    return 0;
}

int svd_serial
(
    double const *A, /* input m-by-n matrx */
    double *Up, /* output m-by-p matrix */
    double *Sp, /* output p-by-p diagonal matrix */
    double *Vtp, /* output p-by-n matrix */
    int m, /* rows of A */
    int n, /* columns of A */
    int p, /* rank approximation */
    int b /* number of seed nodes in binary topology */
)
{
    double *Al = malloc(m*n*sizeof(double));
    memcpy(Al, A, m*n*sizeof(double));

    int q = log2i(b);
    assert(q >= 1);

    int s = n / b;

    if (s <= p)
    {
        fprintf(stderr, "[error] svd_serial: trying to compute %d-truncated SVD with s=%d\n", p, s);
        return -1;
    }

    double *Acat = malloc(m*p*b*sizeof(double));
    double *Vtcat = malloc(p*s*b*sizeof(double)); /* note: s*b == n */

    double const *Ai;
    double *A1i, *Vt1i;

    for (int i = 0; i < b; ++i)
    {
        Ai = &Al[i*m*s];
        A1i = &Acat[i*m*p];
        Vt1i = &Vtcat[i*p*s];

        seed_node(Ai, A1i, Vt1i, m, n, q, p);
    }

    double *Ak_2i_0, *Vtk_2i_0, *Ak_2i_1, *Vtk_2i_1, *Ak1_lj, *Vtk1_lj;

    for (int k = 1; k < q; ++k)
    {
        int c = 1 << (q-k); /* nodes on this level */
        int d = s * (1 << (k-1)); /* column count of incoming Vtk_2i_j matrices */

        for (int i = 0; i < c; ++i)
        {
            Ak_2i_0 = &Acat[(2*i)*m*p];
            Ak_2i_1 = &Acat[(2*i+1)*m*p];
            Vtk_2i_0 = &Vtcat[(2*i)*p*d];
            Vtk_2i_1 = &Vtcat[(2*i+1)*p*d];

            Ak1_lj = &Acat[i*m*p];
            Vtk1_lj = &Vtcat[(2*i)*p*d];

            combine_node(Ak_2i_0, Vtk_2i_0, Ak_2i_1, Vtk_2i_1, Ak1_lj, Vtk1_lj, m, n, k, q, p);
        }
    }

    double *Aq1_11, *Aq1_12, *Vtq1_11, *Vtq1_12;

    Aq1_11 = &Acat[0];
    Aq1_12 = &Acat[m*p];
    Vtq1_11 = &Vtcat[0];
    Vtq1_12 = &Vtcat[(n*p)>>1];

    extract_node(Aq1_11, Vtq1_11, Aq1_12, Vtq1_12, Up, Sp, Vtp, m, n, q, p);

    free(Acat);
    free(Vtcat);
    free(Al);

    return 0;
}

double get_clock_telapsed(struct timespec start, struct timespec end)
{
    uint64_t diff = 1000000000L * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec);
    return (diff / 1000000000.0);
}
