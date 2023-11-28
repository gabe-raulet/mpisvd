#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "cblas.h"
#include "lapacke.h"
#include "svd_routines.h"

int log2i(int v)
{
    int x = 0;
    while (v >>= 1) ++x;
    return x;
}

/*
 * Computes an approximate p-truncated SVD A = Up*Sp*Vtp.
 */
int svd_serial
(
    double const *A, /* input m-by-n matrx */
    double **Up, /* output m-by-p matrix */
    double **Sp, /* output p-by-p diagonal matrix */
    double **Vtp, /* output p-by-n matrix */
    int m, /* rows of A */
    int n, /* columns of A */
    int p, /* rank approximation */
    int b /* number of seed nodes in binary topology */
)
{
    if (!A || !Up || !Sp || !Vtp)
    {
        fprintf(stderr, "[error] svd_serial: invalid arguments\n");
        return -1;
    }

    if (m < n || n <= 0 || p > n || b <= 0)
    {
        fprintf(stderr, "[error] svd_serial: must have 1 <= p <= n <= m and b >= 1.\n[note]: support for m < n would be nice at some point\n");
        return -1;
    }

    if ((m&(m-1)) || (n&(n-1)) || ((b&(b-1))))
    {
        fprintf(stderr, "[error] svd_serial: currently only works with m, n, and b being powers of 2\n");
        return -1;
    }

    if (b >= n || n % b != 0)
    {
        fprintf(stderr, "[error] svd_serial: because of even column-splitting requirement, it is necessary that n %% b == 0 and n > b\n");
        return -1;
    }

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
        Ai = &A[i*m*s];
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

    return 0;
}
