#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "kiss.h"
#include "cblas.h"
#include "lapacke.h"
#include "mmio_dense.h"
#include "linalg_routines.h"
#include "svd_routines.h"

int iseed[4];

int iseed_init();
int log2i(int v);

int main(int argc, char *argv[])
{
    kiss_init();
    iseed_init();

    int q = 3;   /* 2^q is number of compute nodes */
    int m = 256; /* number of rows of global matrix A */
    int n = 128; /* number of columns of global matrix A */
    int r = 128; /* rank of matrix A */
    double cond = 100.; /* condition number of A */
    double damping = 2.; /* damping factor */

    if (m < n || n < r || r < 1 || cond <= 0 || damping < 1)
    {
        fprintf(stderr, "Error: invalid parameters\n");
        return 1;
    }

    int p = 10; /* number of singular values/vectors of A we are computing */
    int b = 1 << q; /* number of seed nodes (==nprocs) */
    int s = n / b; /* number of columns for each local submatrix partition of A at start of algorithm */

    if (s < p)
    {
        fprintf(stderr, "Error: fewer than p=%d columns per seed node; should run with fewer tasks\n", p);
        return 1;
    }

    double *A, *S;
    FILE *f;

    A = malloc(m*n*sizeof(double));
    S = calloc(r, sizeof(double));

    /*
     * Singular values of A will be S[0]=cond, S[1]=cond/damping, S[2]=cond/(damping^2), ..., S[r-1] = cond/(damping^(r-1)).
     */

    S[0] = cond;
    for (int i = 1; i < r; ++i)
        S[i] = S[i-1] / damping;

    /*
     * DLATMS generates a matrix A with specified singular values.
     */
    LAPACKE_dlatms(LAPACK_COL_MAJOR, m, n, 'U', iseed, 'N', S, 0, 0., 0., m, n, 'N', A, m);

    f = fopen("A.mtx", "w");
    mmio_write_dense(f, A, m, n, 0);
    fclose(f);

    double *Akisflat, *Vtkisflat, **A1is, **Akis, **Vtis, **Vtkis;
    (void)Vtis;

    Akisflat = malloc(m*p*b*sizeof(double));
    Vtkisflat = malloc(n*p*sizeof(double));

    A1is = malloc(b*sizeof(double*));
    Akis = malloc(b*sizeof(double*));
    Vtkis = malloc(b*sizeof(double*));

    /*
     * 1. Each matrix pointed to by A1kis is m-by-s
     * 2. Each matrix pointed to by Akis is m-by-p
     * 3. Each matrix pointed to by Vtkis is p-by-s
     */

    for (int i = 0; i < b; ++i)
    {
        A1is[i] = &A[i*m*s];
        Akis[i] = &Akisflat[i*m*p];
        Vtkis[i] = &Vtkisflat[i*p*s];
    }

    for (int i = 0; i < b; ++i)
    {
        seed_node(A1is[i], Akis[i], Vtkis[i], m, s, p);
    }

    for (int l = 1; l <= q; ++l)
    {
        int cur_node_cnt = 1 << (q-l);
        /*int prev_node_cnt = cur_node_cnt << 1;*/

        for (int i = 0; i < cur_node_cnt; ++i)
        {
            combine_node(Akis[2*i], Vtkis[2*i], l, m, s, p);
        }
    }

    f = fopen("A11.mtx", "w");
    mmio_write_dense(f, Akis[0], m, p, 0);
    fclose(f);

    f = fopen("A12.mtx", "w");
    mmio_write_dense(f, Akis[1], m, p, 0);
    fclose(f);

    f = fopen("V11.mtx", "w");
    mmio_write_dense(f, Vtkis[0], p, n/2, 0);
    fclose(f);

    f = fopen("V12.mtx", "w");
    mmio_write_dense(f, Vtkis[1], p, n/2, 0);
    fclose(f);

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
    iseed[3] |= 1; /* iseed[3] must be odd */
    return 0;
}
