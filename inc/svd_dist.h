#ifndef SVD_MPI_H_
#define SVD_MPI_H_

#include <mpi.h>

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
);

int svd_dist(const double *Aloc, double **Up, double **Sp, double **Vtp, int m, int n, int p, int root, MPI_Comm comm);


#endif
