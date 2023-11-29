#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "kiss.h"
#include "cblas.h"
#include "lapacke.h"
#include "mmio_dense.h"

int iseed[4];

int iseed_init();
int cmpfnc(const void *a, const void *b);
int svd_test_gen_direct(double *A, double const *SIGMA, int m, int n);

int main(int argc, char *argv[])
{
    kiss_init();
    iseed_init();

    int m = 256;
    int n = 128;
    int r = 64;

    double *A = (double*) malloc(m*n*sizeof(double));
    double *S = (double*) calloc(n, sizeof(double));

    S[0] = 100.;
    for (int i = 1; i < r; ++i) S[i] = S[i-1]/2.;

    svd_test_gen_direct(A, S, m, n);
    mmio_write_dense(stdout, A, m, n, 0);

    free(A);
    free(S);

    return 0;
}

int cmpfnc(const void *a, const void *b)
{
    double A = *((double *)a);
    double B = *((double *)b);

    if      (A < B) return  1;
    else if (A > B) return -1;
    else            return  0;
}

int iseed_init()
{
    for (int i = 0; i < 4; ++i) iseed[i] = (kiss_rand() % 4096);
    iseed[3] &= (iseed[3]^1); /* iseed[3] must be odd */
    return 0;
}

int svd_test_gen_direct(double *A, double const *SIGMA, int m, int n)
{
    /*
     * Generate a random m-by-n matrix with singular values in SIGMA[0..min(m,n)].
     */

    assert(A != NULL & SIGMA != NULL && m >= 1 && n >= 1);

    double *D = (double *)SIGMA;
    LAPACKE_dlatms(LAPACK_COL_MAJOR, m, n, 'U', iseed, 'N', D, 0, 0., 0., m, n, 'N', A, m);

    return 0;
}
