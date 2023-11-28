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
}

///*
// * Computes an approximate p-truncated SVD A = Up*Sp*Vtp.
// */
//int svd_serial
//(
//    double const *A, /* input m-by-n matrx */
//    double **Up, /* output m-by-p matrix */
//    double **Sp, /* output p-by-p diagonal matrix */
//    double **Vtp, /* output p-by-n matrix */
//    int m, /* rows of A */
//    int n, /* columns of A */
//    int p, /* rank approximation */
//    int b /* number of seed nodes in binary topology */
//)
//{
//    if (!A || !Up || !Sp || !Vtp)
//    {
//        fprintf(stderr, "[error] svd_serial: invalid arguments\n");
//        return -1;
//    }
//
//    if (m < n || n <= 0 || p > n || b <= 0)
//    {
//        fprintf(stderr, "[error] svd_serial: must have 1 <= p <= n <= m and b >= 1.\n[note]: support for m < n would be nice at some point\n");
//        return -1;
//    }
//
//    if ((m&(m-1)) || (n&(n-1)) || ((b&(b-1))))
//    {
//        fprintf(stderr, "[error] svd_serial: currently only works with m, n, and b being powers of 2\n");
//        return -1;
//    }
//
//    if (b >= n || n % b != 0)
//    {
//        fprintf(stderr, "[error] svd_serial: because of even column-splitting requirement, it is necessary that n %% b == 0 and n > b\n");
//        return -1;
//    }
//
//    int q = log2i(b);
//    assert(q >= 1);
//
//    int s = n / b;
//
//    if (s <= p)
//    {
//        fprintf(stderr, "[error] svd_serial: trying to compute %d-truncated SVD with s=%d\n", p, s);
//        return -1;
//    }
//
//    double *Acat = malloc(m*p*b*sizeof(double));
//    double *Vtcat = malloc(p*s*b*sizeof(double)); /* note: s*b == n */
//
//    double const *Ai;
//    double *A1i, *Vt1i;
//
//    for (int i = 0; i < b; ++i)
//    {
//        Ai = &A[i*m*s];
//        A1i = &Acat[i*m*p];
//        Vt1i = &Vtcat[i*p*s];
//
//        seed_node(Ai, A1i, Vt1i, m, n, q, p);
//    }
//
//    double *Ak_2i_0, *Vtk_2i_0, *Ak_2i_1, *Vtk_2i_1, *Ak1_lj, *Vtk1_lj;
//
//    for (int k = 1; k < q; ++k)
//    {
//        int c = 1 << (q-k); /* nodes on this level */
//        int d = s * (1 << (k-1)); /* column count of incoming Vtk_2i_j matrices */
//
//        for (int i = 0; i < c; ++i)
//        {
//            Ak_2i_0 = &Acat[(2*i)*m*p];
//            Ak_2i_1 = &Acat[(2*i+1)*m*p];
//            Vtk_2i_0 = &Vtcat[(2*i)*p*d];
//            Vtk_2i_1 = &Vtcat[(2*i+1)*p*d];
//
//            Ak1_lj = &Acat[i*m*p];
//            Vtk1_lj = &Vtcat[(2*i)*p*d];
//
//            combine_node(Ak_2i_0, Vtk_2i_0, Ak_2i_1, Vtk_2i_1, Ak1_lj, Vtk1_lj, m, n, k, q, p);
//        }
//    }
//
//    double *Aq1_11, *Aq1_12, *Vtq1_11, *Vtq1_12;
//
//    Aq1_11 = &Acat[0];
//    Aq1_12 = &Acat[m*p];
//    Vtq1_11 = &Vtcat[0];
//    Vtq1_12 = &Vtcat[(n*p)>>1];
//
//    extract_node(Aq1_11, Vtq1_11, Aq1_12, Vtq1_12, Up, Sp, Vtp, m, n, q, p);
//
//    free(Acat);
//    free(Vtcat);
//
//    return 0;
//}
