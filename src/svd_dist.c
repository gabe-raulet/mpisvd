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

    /*char fname[1024];*/
    /*snprintf(fname, 1024, "A1%d.mtx", myrank);*/
    /*mmwrite(fname, A1i, m, s);*/
    /*snprintf(fname, 1024, "Vt1%d.mtx", myrank);*/
    /*mmwrite(fname, Vt1i, s, p);*/

    /*
     * A1i is m-by-p
     * Vt1i is p-by-s
     */

    double *Arecv = malloc(2*m*p*sizeof(double));
    double *Vtrecv = malloc(n*p*sizeof(double)); /* this should be allocated with less memory depending on what myrank is */

    /* receive from seeds */

    //if (myrank % 2 != 0)
    //{
    //    MPI_Send(A1i, m*s, MPI_DOUBLE, myrank-1, myrank, comm);
    //    MPI_Send(Vt1i, s*p, MPI_DOUBLE, myrank-1, myrank+nprocs, comm);
    //}
    //else
    //{
    //    memcpy(Arecv, A1i, m*s*sizeof(double));
    //    memcpy(Vtrecv, Vt1i, s*p*sizeof(double));
    //    MPI_Recv(&Arecv[m*s], m*s, MPI_DOUBLE, myrank+1, myrank+1, comm);
    //    MPI_Recv(&Vtrecv[s*p], s*p, MPI_DOUBLE, myrank+1, myrank+1+nprocs, comm);
    //}

    //double *Ak_2i_0, *Vtk_2i_0, *Ak_2i_1, *Vtk_2i_1, *Ak1_lj, *Vtk1_lj;

    //for (int k = 1; k < q; ++k)
    //{
    //    int c = 1 << (q-k); /* nodes on this level */
    //    int d = s * (1 << (k-1)); /* column count of incoming Vtk_2i_j matrices */

    //    Ak_2i_0 = &Arecv[0];
    //    Ak_2i_1 = &Arecv[m*p];
    //    Vtk_2i_0 = &Vtrecv[0];
    //    Vtk_2i_1 = &Vtrecv[p*d];

    //    Ak1_lj = &Arecv[0];
    //    Vtk1_lj = &Vtrecv[]


    //    //for (int i = 0; i < c; ++i)
    //    //{
    //    //    Ak_2i_0 = &Acat[(2*i)*m*p];
    //    //    Ak_2i_1 = &Acat[(2*i+1)*m*p];
    //    //    Vtk_2i_0 = &Vtcat[(2*i)*p*d];
    //    //    Vtk_2i_1 = &Vtcat[(2*i+1)*p*d];

    //    //    Ak1_lj = &Acat[i*m*p];
    //    //    Vtk1_lj = &Vtcat[(2*i)*p*d];

    //    //    combine_node(Ak_2i_0, Vtk_2i_0, Ak_2i_1, Vtk_2i_1, Ak1_lj, Vtk1_lj, m, n, k, q, p);
    //    //}
    //}

    free(A1i);
    free(Vt1i);
    MPI_Barrier(comm);
    return 0;
}

