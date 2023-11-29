#include "mmio_dense.h"
#include "mmio.h"
#include <stdlib.h>
#include <assert.h>

int mmio_read_dense(FILE *f, double **A_ref, int *M, int *N, int row_major)
{
    assert(f != NULL && A_ref != NULL && M != NULL && N != NULL);

    MM_typecode matcode;
    int m, n, i, j;
    double *A;

    mm_read_banner(f, &matcode);
    assert(mm_is_dense(matcode));
    mm_read_mtx_array_size(f, &m, &n);
    A = (double*) malloc(m*n*sizeof(double));
    assert(A != NULL);

    if (!row_major)
    {
        for (i = 0; i < m*n; ++i)
            fscanf(f, "%lg", &A[i]);
    }
    else
    {
        for (j = 0; j < n; ++j)
            for (i = 0; i < m; ++i)
                fscanf(f, "%lg", &A[i*n + j]);
    }

    *A_ref = A;
    *M = m;
    *N = n;

    return 0;
}

int mmio_write_dense(FILE *f, double const *A, int m, int n, int row_major)
{
    assert(f != NULL && A != NULL && m >= 1 && n >= 1);

    MM_typecode matcode;
    int i, j;

    mm_initialize_typecode(&matcode);
    mm_set_matrix(&matcode);
    mm_set_dense(&matcode);
    mm_set_real(&matcode);
    mm_set_general(&matcode);

    mm_write_banner(f, matcode);
    fprintf(f, "%%\n");
    mm_write_mtx_array_size(f, m, n);

    if (!row_major)
    {
        for (i = 0; i < m*n; ++i)
            fprintf(f, "%.18e\n", A[i]);
    }
    else
    {
        for (j = 0; j < n; ++j)
            for (i = 0; i < m; ++i)
                fprintf(f, "%.18e\n", A[i*n + j]);
    }

    return 0;
}

int mmwrite_diagonal(char const *fname, double const *D, int n)
{
    FILE *f;
    MM_typecode matcode;

    f = fopen(fname, "w");

    mm_initialize_typecode(&matcode);
    mm_set_matrix(&matcode);
    mm_set_dense(&matcode);
    mm_set_real(&matcode);
    mm_set_general(&matcode);

    mm_write_banner(f, matcode);
    fprintf(f, "%%\n");
    mm_write_mtx_array_size(f, n, n);

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
        {
            double v = i != j? 0.0 : D[i];
            fprintf(f, "%.18e\n", v);
        }

    fclose(f);

    return 0;
}

int mmwrite_upper_triangular(char const *fname, double const *A, int n, int lda)
{
    FILE *f;
    MM_typecode matcode;

    f = fopen(fname, "w");

    mm_initialize_typecode(&matcode);
    mm_set_matrix(&matcode);
    mm_set_dense(&matcode);
    mm_set_real(&matcode);
    mm_set_general(&matcode);

    mm_write_banner(f, matcode);
    fprintf(f, "%%\n");
    mm_write_mtx_array_size(f, n, n);

    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i <= j; ++i)
        {
            fprintf(f, "%.18e\n", A[i + j*lda]);
        }

        for (int i = j+1; i < n; ++i)
        {
            fprintf(f, "%.18e\n", 0.0);
        }
    }

    fclose(f);
    return 0;
}

double* mmread(char const *fname, int *m, int *n)
{
    double *A;
    FILE *f;

    f = fopen(fname, "r");
    mmio_read_dense(f, &A, m, n, 0);
    fclose(f);

    return A;
}

int mmwrite(char const *fname, double const *A, int m, int n)
{
    FILE *f;

    f = fopen(fname, "w");
    mmio_write_dense(f, A, m, n, 0);
    fclose(f);

    return 0;
}

int write_diag(const char *fname, const double *D, int n)
{
    FILE *f = fopen(fname, "w");
    for (int i = 0; i < n; ++i)
        fprintf(f, "%.18e\n", D[i]);
    fclose(f);
    return 0;
}
