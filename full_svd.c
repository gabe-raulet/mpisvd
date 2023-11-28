#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "kiss.h"
#include "cblas.h"
#include "lapacke.h"
#include "mmio_dense.h"
#include "svd_routines.h"

int iseed[4];

int iseed_init();
int log2i(int v);

int main(int argc, char *argv[])
{
    kiss_init();
    iseed_init();

    int m, n, p, q, s, b;
    double *A;

    if (argc != 4)
    {
        fprintf(stderr, "usage: %s <A.mtx> <p> <q>\n", argv[0]);
        return 1;
    }

    A = mmread(argv[1], &m, &n);

    assert(!(m&(m-1)) && !(n&(n-1))); /* m and n should be powers of 2 */
    assert(m >= n); /* temporary */

    p = atoi(argv[2]); /* p-truncated svd */
    q = atoi(argv[3]); /* 2^q nodes */

    assert(1 <= p && p <= n); /* n = min(m,n) */
    assert(q >= 1);

    b = 1 << q; /* number of nodes */
    assert(n > b);

    s = n / b; /* entry column split count */
    assert(s >= p);

    double *Acat, *Vtcat;

    Acat = malloc(m*p*b*sizeof(double));
    Vtcat = malloc(p*s*b*sizeof(double)); /* note: s*b == n */

    double *Ai, *A1i, *Vt1i;

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
    double *Up, *Sp, *Vtp;

    Aq1_11 = &Acat[0];
    Aq1_12 = &Acat[m*p];
    Vtq1_11 = &Vtcat[0];
    Vtq1_12 = &Vtcat[(n*p)>>1];

    extract_node(Aq1_11, Vtq1_11, Aq1_12, Vtq1_12, &Up, &Sp, &Vtp, m, n, q, p);

    mmwrite_diagonal("Sp_a.mtx", Sp, p);
    mmwrite("Up_a.mtx", Up, m, p);
    mmwrite("Vtp_a.mtx", Vtp, p, n);

    free(A);
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
