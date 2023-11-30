#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <math.h>
#include "kiss.h"
#include "cblas.h"
#include "lapacke.h"
#include "mmio_dense.h"

/*# compute UU^T or L^TL*/
/*[sdcz] lauum ( uplo, n, A, ldA, info )*/

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

int main(int argc, char *argv[])
{
    params_t params;

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

    double *A = malloc(m*n*sizeof(double));
    double *S = malloc(n*sizeof(double));

    gen_test_mat(A, S, m, n, mode, cond, dmax);

    double *Scheck = malloc(n*sizeof(double));
    double *U = malloc(m*n*sizeof(double));
    double *Vt = malloc(n*n*sizeof(double));

    gen_uv_mats(A, Scheck, U, Vt, m, n);

    fprintf(stderr, "[main] l2-distance between DLATMS and DGESVD singular values = %.18e\n", l2dist(S, Scheck, n));

    free(A);
    free(S);

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
