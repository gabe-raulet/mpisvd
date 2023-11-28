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

    f = fopen(argv[1]);
    mmio_read_dense(f, &A, &m, &n, 0);
    fclose(f);

    p = 10, q = 5; /* must match seed_test.py */

    param_check_and_get(m, n, q, p, &b, &s);

    double *Ais = malloc(m*s*b*sizeof(double));
    double *Acat = malloc(m*p*b*sizeof(double));
    double *Vtcat = malloc(p*s*b*sizeof(double));

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
