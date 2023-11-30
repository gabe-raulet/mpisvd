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

typedef struct { int m, n, mode, seed; double cond, dmax; char *label; } params_t;

int iseed[4];
int iseed_init();
int log2i(int v);

int svdgen_param_check(int m, int n, int mode, double cond, double dmax);
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

    int m=params.m, n=params.n, mode=params.mode;
    double cond=params.cond, dmax=params.dmax;

    if (svdgen_param_check(m, n, mode, cond, dmax))
        return 1;

    double *A = malloc(m*n*sizeof(double));
    double *S = malloc(n*sizeof(double));

    gen_test_mat(A, S, m, n, mode, cond, dmax);

    double *Scheck = malloc(n*sizeof(double));
    double *U = malloc(m*n*sizeof(double));
    double *Vt = malloc(n*n*sizeof(double));

    gen_uv_mats(A, Scheck, U, Vt, m, n);

    int show = n < 5? n : 5;
    fprintf(stderr, "[main:gen_truth] diag(S)[0..%d] = ", show-1);
    for (int i = 0; i < show; ++i) fprintf(stderr, "%.3e,", S[i]);
    fprintf(stderr, "%s\n", n < 5? "" : "...");

    fprintf(stderr, "[main:gen_truth] err=%.18e (DLATMS-vs-DGESVD) [S :: singular values]\n", l2dist(S, Scheck, n));
    free(Scheck);

    char fname[1024];

    snprintf(fname, 1024, "A_%s.mtx", params.label);
    mmwrite(fname, A, m, n);

    snprintf(fname, 1024, "U_%s.mtx", params.label);
    mmwrite(fname, U, m, n);

    snprintf(fname, 1024, "Vt_%s.mtx", params.label);
    mmwrite(fname, Vt, n, n);

    snprintf(fname, 1024, "S_%s.diag", params.label);

    FILE *f = fopen(fname, "w");
    for (int i = 0; i < n; ++i)
        fprintf(f, "%.18e\n", S[i]);
    fclose(f);

    free(A);
    free(S);
    free(U);
    free(Vt);

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
    ps->mode = -1;
    ps->cond = 100.f;
    ps->dmax = 2.f;
    ps->seed = -1;
    ps->label = NULL;

    return 0;
}

int usage(char *argv[], const params_t *ps)
{
    fprintf(stderr, "Usage: %s [options]\n", argv[0]);
    fprintf(stderr, "Options: -m INT    rows of test matrix [%d]\n", ps->m);
    fprintf(stderr, "         -n INT    columns of test matrix [%d]\n", ps->n);
    fprintf(stderr, "         -u INT    matrix generation mode [%d]\n", ps->mode);
    fprintf(stderr, "         -c FLOAT  DLATMS cond or condition number [%.3e]\n", ps->cond);
    fprintf(stderr, "         -d FLOAT  DLATMS dmax or damping factor [%.3e]\n", ps->dmax);
    fprintf(stderr, "         -s INT    RNG seed [%d]\n", ps->seed);
    fprintf(stderr, "         -o STR    output label\n");
    fprintf(stderr, "         -h        help message\n");
    return 1;
}

int parse_params(int argc, char *argv[], params_t *ps)
{
    int c;
    params_init(ps);

    while ((c = getopt(argc, argv, "m:n:u:c:d:s:o:h")) >= 0)
    {
        if      (c == 'm') ps->m = atoi(optarg);
        else if (c == 'n') ps->n = atoi(optarg);
        else if (c == 'u') ps->mode = atoi(optarg);
        else if (c == 'c') ps->cond = atof(optarg);
        else if (c == 'd') ps->dmax = atof(optarg);
        else if (c == 's') ps->seed = atoi(optarg);
        else if (c == 'o') ps->label = optarg;
        else if (c == 'h')
        {
            params_init(ps);
            return usage(argv, ps);
        }
    }

    if (!ps->label)
    {
        fprintf(stderr, "error: missing required -o parameter\n");
        return -1;
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
