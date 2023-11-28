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
void param_check_and_get(int m, int n, int q, int p, int *b, int *s);

int main(int argc, char *argv[])
{
    kiss_init();
    iseed_init();

    int m, n, q, p, b, s;
    double *A;
    FILE *f;

    f = fopen(argv[1], "r");
    mmio_read_dense(f, &A, &m, &n, 0);
    fclose(f);

    p = atoi(argv[2]);
    q = atoi(argv[3]);
    param_check_and_get(m, n, q, p, &b, &s);

    double *Acat = malloc(m*p*b*sizeof(double));
    double *Vtcat = malloc(p*s*b*sizeof(double)); /* note: s*b == n */

    double const *Ai;
    double *A1i;
    double *Vt1i;

    for (int i = 0; i < b; ++i)
    {
        Ai = &A[i*m*s];
        A1i = &Acat[i*m*p]; printf("A1%d = &Acat[%d*%d*%d] = &Acat[%d]\n", i, i, m, p, i*m*p);
        Vt1i = &Vtcat[i*p*s]; printf("Vt1%d = &Vtcat[%d*%d*%d] = &Vtcat[%d]\n", i, i, p, s, i*p*s);

        seed_node(Ai, A1i, Vt1i, m, n, q, p);
    }

    f = fopen("Aseed_test.mtx", "w");
    mmio_write_dense(f, Acat, m, p*b, 0);
    fclose(f);

    f = fopen("Vtseed_test.mtx", "w");
    mmio_write_dense(f, Vtcat, p, s*b, 0);
    fclose(f);

    //for (int k = 1; k < q; ++k)
    //{
    //    int c = 1 << (q-k);
    //    int d = s * (1 << (k-1));

    //    for (int i = 0; i < c; ++i)
    //    {
    //        double *Ak_2i_0; /* m-by-p */
    //        double *Ak_2i_1; /* m-by-p */
    //        double *Vtk_2i_0; /* p-by-d */
    //        double *Vtk_2i_1; /* p-by-d */
    //        double *Ak1_lj; /* m-by-2p */
    //        double *Vtk1_lj; /* p-by-2d; d := s * (2^(k-1)) */

    //        Ak_2i_0 = &Acat[(2*i)*m*p]; printf("A%d_2%d_0 = &Acat[%d*%d*%d] = &Acat[%d]\n", k, i, 2*i, m, p, 2*i*m*p);
    //        Ak_2i_1 = &Acat[(2*i+1)*m*p]; printf("A%d_2%d_1 = &Acat[%d*%d*%d] = &Acat[%d]\n", k, i, 2*i+1, m, p, (2*i+1)*m*p);
    //        Vtk_2i_0 = &Vtcat[(2*i)*d*s]; printf("Vt%d_2%d_0 = &Vtcat[%d*%d*%d] = &Vtcat[%d]\n", k, i, 2*i, d, s, 2*i*d*s);
    //        Vtk_2i_1 = &Vtcat[(2*i+1)*d*s]; printf("Vt%d_2%d_1 = &Vtcat[%d*%d*%d] = &Vtcat[%d]\n", k, i, 2*i+1, d, s, (2*i+1)*d*s);

    //        Ak1_lj = &Acat[i*m*p];
    //        Vtk1_lj = &Vtcat[i*p*(2*d)];

    //        combine_node(Ak_2i_0, Vtk_2i_0, Ak_2i_1, Vtk_2i_1, Ak1_lj, Vtk1_lj, m, n, k, q, p);
    //    }
    //}

    f = fopen("Acombs_test.mtx", "w");
    mmio_write_dense(f, Acat, m, p*2, 0);
    fclose(f);

    f = fopen("Vtcombs_test.mtx", "w");
    mmio_write_dense(f, Vtcat, p, n, 0);
    fclose(f);

    free(Acat);
    free(Vtcat);
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

void param_check_and_get(int m, int n, int q, int p, int *b, int *s)
{
    int _b = 1 << q;
    int _s = n / _b;
    assert(!(m&(m-1)));
    assert(!(n&(n-1)));
    assert(q >= 1);
    assert(n >= _b);
    assert(_s >= p);
    assert(p >= 1);
    *b = _b;
    *s = _s;
}
