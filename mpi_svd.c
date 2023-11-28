#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include "kiss.h"
#include "cblas.h"
#include "lapacke.h"
#include "mmio_dense.h"
#include "svd_routines.h"

int iseed[4];

int iseed_init();
int log2i(int v);

int svd_test_gen
(
    double    **A, /* global m-by-n matrix A returned to root */
    double **Aloc, /* local m-by-(n/nprocs) matrix scattered to everyone */
    double    **U, /* global m-by-n left singular vectors matrix U returned to root */
    double    **S, /* global n-by-n diagonal singular values matrix S returned to root */
    double   **Vt, /* global n-by-n right singular vectors matrix Vt returned to root */
    int        *s, /* local columns of A */
    int         m, /* rows of A */
    int         n, /* columns of A */
    int         r, /* rank of A */
    double   cond, /* condition number of A */
    double   damp, /* singular value damping factor of A */
    int      root, /* root node */
    MPI_Comm comm  /* communicator */
);


int main(int argc, char *argv[])
{
    kiss_init();
    iseed_init();

    int myrank, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int m = 256;
    int n = 256;
    int r = 256;
    double cond = 100.;
    double damp = 2.;
    int s;

    double *A, *Aloc, *U, *S, *Vt;

    svd_test_gen(&A, &Aloc, &U, &S, &Vt, &s, m, n, r, cond, damp, 0, MPI_COMM_WORLD);

    if (!myrank)
    {
        mmwrite("A.mtx", A, m, n);
        mmwrite("U.mtx", U, m, n);
        mmwrite("Vt.mtx", Vt, n, n);
        mmwrite_diagonal("S.mtx", S, n);

        free(A); free(U); free(Vt); free(S);
    }

    char fname[1024];
    snprintf(fname, 1024, "A%d.mtx", myrank);

    mmwrite(fname, Aloc, m, s);

    MPI_Finalize();
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

int svd_test_gen
(
    double    **A, /* global m-by-n matrix A returned to root */
    double **Aloc, /* local m-by-(n/nprocs) matrix scattered to everyone */
    double    **U, /* global m-by-n left singular vectors matrix U returned to root */
    double    **S, /* global n-by-n diagonal singular values matrix S returned to root */
    double   **Vt, /* global n-by-n right singular vectors matrix Vt returned to root */
    int        *s, /* local columns of A */
    int         m, /* rows of A */
    int         n, /* columns of A */
    int         r, /* rank of A */
    double   cond, /* condition number of A */
    double   damp, /* singular value damping factor of A */
    int      root, /* root node */
    MPI_Comm comm  /* communicator */
)
{
    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    double *Aroot, *Apart, *Uroot, *Sroot, *Vtroot;

    assert(m >= n && n >= 1 && n % nprocs == 0);

    if (myrank == root)
    {
        Aroot = malloc(m*n*sizeof(double));
        Sroot = calloc(n, sizeof(double));
        Uroot = malloc(m*n*sizeof(double));
        Vtroot = malloc(n*n*sizeof(double));

        double *work = malloc(5*n*sizeof(double));

        Sroot[0] = cond;

        for (int i = 1; i < r; ++i)
            Sroot[i] = Sroot[i-1] / damp;

        LAPACKE_dlatms(LAPACK_COL_MAJOR, m, n, 'U', iseed, 'N', Sroot, 0, 0., 0., m, n, 'N', Aroot, m);
        LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'S', m, n, Aroot, m, Sroot, Uroot, m, Vtroot, n, work);
        free(work);
    }

    int nloc = n / nprocs;
    Apart = malloc(m*nloc*sizeof(double));

    MPI_Scatter(Aroot, m*nloc, MPI_DOUBLE, Apart, m*nloc, MPI_DOUBLE, root, comm);

    *s = nloc;
    *Aloc = Apart;

    if (myrank == root)
    {
        *A = Aroot;
        *U = Uroot;
        *S = Sroot;
        *Vt = Vtroot;
    }
    else
    {
        *U = *S = *Vt = NULL;
    }

    return 0;
}
