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
#include "svd_dist.h"
#include "svd_serial.h"

int generate_svd_dist_test
(
    double    **A, /* global m-by-n matrix A returned to root */
    double **Aloc, /* local m-by-(n/nprocs) matrix scattered to everyone */
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

    double *Aroot, *Apart;

    if (myrank == root)
    {
        if (generate_svd_test(&Aroot, m, n, r, cond, damp) != 0)
            MPI_Abort(comm, 1);
    }

    assert(n % nprocs == 0);

    int nloc = n / nprocs;

    Apart = malloc(m*nloc*sizeof(double));
    MPI_Scatter(Aroot, m*nloc, MPI_DOUBLE, Apart, m*nloc, MPI_DOUBLE, root, comm);

    *s = nloc;
    *Aloc = Apart;

    MPI_Barrier(comm);
    return 0;
}
