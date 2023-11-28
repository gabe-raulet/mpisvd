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

double* quick_read(char const *fname, int *m, int *n)
{
    double *A;
    FILE *f = fopen(fname, "r");
    mmio_read_dense(f, &A, m, n, 0);
    fclose(f);
    return A;
}

void quick_write(char const *fname, double *A, int m, int n)
{
    FILE *f = fopen(fname, "w");
    mmio_write_dense(f, A, m, n, 0);
    fclose(f);
}

int main(int argc, char *argv[])
{
    kiss_init();
    iseed_init();

    int m, n, p, d;
    double *A, *Ak_2i_0, *Ak_2i_1, *Vtk_2i_0, *Vtk_2i_1, *Ak1_lj, *Vtk1_lj;

    int tmp, tmp2;
    A = quick_read("A.mtx", &m, &n);
    free(A);

    Ak_2i_0 = quick_read("Ak_2i_0.mtx", &tmp, &p); /* m-by-p */
    assert(tmp == m);

    Ak_2i_1 = quick_read("Ak_2i_1.mtx", &tmp, &tmp2); /* m-by-p */
    assert(tmp == m && tmp2 == p);

    Vtk_2i_0 = quick_read("Vtk_2i_0.mtx", &tmp, &d); /* p-by-d */
    assert(tmp == p);

    Vtk_2i_1 = quick_read("Vtk_2i_1.mtx", &tmp, &tmp2); /* p-by-d */
    assert(tmp == p);

    Ak1_lj = malloc(m*p*sizeof(double));
    Vtk1_lj = malloc(p*(2*d)*sizeof(double));

    combine_node(Ak_2i_0, Vtk_2i_0, Ak_2i_1, Vtk_2i_1, Ak1_lj, Vtk1_lj, m, n, 1, 3, p);

    quick_write("Ak1_lj_attempt.mtx", Ak1_lj, m, p);
    quick_write("Vtk1_lj_attempt.mtx", Vtk1_lj, p, 2*d);

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
