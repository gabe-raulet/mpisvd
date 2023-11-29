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
#include "svd_routines.h"

extern int log2i(int v);

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
    *A = (myrank == root)? Aroot : NULL;

    MPI_Barrier(comm);
    return 0;
}

int svd_dist(const double *Aloc, double **Up, double **Sp, double **Vtp, int m, int n, int p, int root, MPI_Comm comm)
{
    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    assert(n % nprocs == 0);

    int s = n / nprocs;

    double *A1i, *Vt1i;

    A1i = malloc(m*s*sizeof(double));
    Vt1i = malloc(s*p*sizeof(double));

    int q = log2i(nprocs);

    seed_node(Aloc, A1i, Vt1i, m, n, q, p);

    double *Amem = malloc(2*m*p*sizeof(double));
    double *Vtmem = malloc(n*p*sizeof(double)); /* this should be allocated with less memory depending on what myrank is */

    if (myrank % 2 != 0)
    {
        MPI_Send(A1i, m*p, MPI_DOUBLE, myrank-1, myrank, comm);
        MPI_Send(Vt1i, p*s, MPI_DOUBLE, myrank-1, myrank+nprocs, comm);
    }
    else
    {
        memcpy(Amem, A1i, m*p*sizeof(double));
        memcpy(Vtmem, Vt1i, p*s*sizeof(double));
        MPI_Recv(&Amem[m*p], m*p, MPI_DOUBLE, myrank+1, myrank+1, comm, MPI_STATUS_IGNORE);
        MPI_Recv(&Vtmem[p*s], p*s, MPI_DOUBLE, myrank+1, myrank+1+nprocs, comm, MPI_STATUS_IGNORE);
    }

    double *Ak_2i_0, *Vtk_2i_0, *Ak_2i_1, *Vtk_2i_1, *Ak1_lj, *Vtk1_lj;

    for (int k = 1; k < q; ++k)
    {
        int d = s * (1 << (k-1)); /* column count of incoming Vtk_2i_j matrices */

        Ak_2i_0 = &Amem[0]; /* m-by-p */
        Ak_2i_1 = &Amem[m*p]; /* m-by-p */
        Vtk_2i_0 = &Vtmem[0]; /* p-by-d */
        Vtk_2i_1 = &Vtmem[p*d]; /* p-by-d */

        Ak1_lj = &Amem[0]; /* m-by-p on exit */
        Vtk1_lj = &Vtmem[0]; /* p-by-2d on exit */

        combine_node(Ak_2i_0, Vtk_2i_0, Ak_2i_1, Vtk_2i_1, Ak1_lj, Vtk1_lj, m, n, k, q, p);

        /* MPI_Recv(buf, count, dtype, source, tag, comm) */
        /* MPI_Send(buf, count, dtype, dest, tag, comm) */

        if ((myrank % (1 << (k+1))) == (1 << k))
        {

            int dest = myrank - (1 << k);
            int Atag = myrank;
            int Vtag = myrank + nprocs;

            /*printf("k=%d: rank %d sending %d doubles to rank %d with tag %d\n", k, myrank, m*p, dest, Atag);*/
            /*printf("k=%d: rank %d sending %d doubles to rank %d with tag %d\n", k, myrank, p*2*d, dest, Vtag);*/

            MPI_Send(Ak1_lj, m*p, MPI_DOUBLE, dest, Atag, comm);
            MPI_Send(Vtk1_lj, p*2*d, MPI_DOUBLE, dest, Vtag, comm);
        }
        else if ((myrank % (1 << (k+1))) == 0)
        {

            int source = myrank + (1 << k);
            int Atag = myrank + (1 << k);
            int Vtag = myrank + (1 << k) + nprocs;

            MPI_Recv(&Amem[m*p], m*p, MPI_DOUBLE, source, Atag, comm, MPI_STATUS_IGNORE);
            MPI_Recv(&Vtmem[p*2*d], p*2*d, MPI_DOUBLE, source, Vtag, comm, MPI_STATUS_IGNORE);

            /*printf("k=%d: rank %d received %d doubles from rank %d with tag %d\n", k, myrank, m*p, source, Atag);*/
            /*printf("k=%d: rank %d received %d doubles from rank %d with tag %d\n", k, myrank, p*2*d, source, Vtag);*/
        }

    }

    MPI_Barrier(comm);

    double *Aq1_11, *Aq1_12, *Vtq1_11, *Vtq1_12;

    Aq1_11 = &Amem[0];
    Aq1_12 = &Amem[m*p];
    Vtq1_11 = &Vtmem[0];
    Vtq1_12 = &Vtmem[(n*p)>>1];

    extract_node(Aq1_11, Vtq1_11, Aq1_12, Vtq1_12, Up, Sp, Vtp, m, n, q, p);

    free(A1i);
    free(Vt1i);
    return 0;
}

