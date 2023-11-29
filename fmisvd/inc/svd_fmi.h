#pragma once

#include <fmi.h>

int log2i(int v);

int generate_svd_fmi_test
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
    int    myrank,
    int    nprocs,
    FMI::Communicator& comm
);

int svd_fmi
(
    const double *Aloc,
    double **Up, 
    double **Sp, 
    double **Vtp, 
    int m, 
    int n, 
    int p, 
    int root, 
    int myrank,
    int nprocs,
    FMI::Communicator& comm
);