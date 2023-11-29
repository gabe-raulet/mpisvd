#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "kiss.h"
#include "cblas.h"
#include "lapacke.h"
#include "mmio_dense.h"
#include "svd_serial.h"
#include "svd_routines.h"

int main(int argc, char *argv[])
{
    if (argc != 9)
    {
        fprintf(stderr, "usage: %s <outprefix> <m> <n> <p> <q> <r> <cond> <damp>\n", argv[0]);
        return 1;
    }

    char const *outprefix = argv[1];
    int m = atoi(argv[2]);
    int n = atoi(argv[3]);
    int p = atoi(argv[4]);
    int q = atoi(argv[5]);
    int r = atoi(argv[6]);
    double cond = atof(argv[7]);
    double damp = atof(argv[8]);

    double *A, *Up, *Sp, *Vtp;

    if (generate_svd_test(&A, m, n, r, cond, damp))
    {
        fprintf(stderr, "error: generate_svd_test\n");
        return 1;
    }

    char fname[1024];

    snprintf(fname, 1024, "A_%s.mtx", outprefix);
    mmwrite(fname, A, m, n);

    if (svd_serial(A, &Up, &Sp, &Vtp, m, n, p, (1<<q)) != 0)
        return 1;

    snprintf(fname, 1024, "Up_%s.mtx", outprefix);
    mmwrite(fname, Up, m, p);

    snprintf(fname, 1024, "Vtp_%s.mtx", outprefix);
    mmwrite(fname, Vtp, p, n);

    snprintf(fname, 1024, "Sp_%s.diag", outprefix);
    write_diag(fname, Sp, p);

    free(A);
    free(Up);
    free(Sp);
    free(Vtp);

    return 0;
}
