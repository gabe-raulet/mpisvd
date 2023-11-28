#ifndef MMIO_DENSE_H_
#define MMIO_DENSE_H_

#include <stdio.h>

int mmio_read_dense(FILE *f, double **A_ref, int *M, int *N, int row_major);
int mmio_write_dense(FILE *f, double const *A, int m, int n, int row_major);

double* mmread(char const *fname, int *m, int *n);
int mmwrite(char const *fname, double const *A, int m, int n);

#endif
