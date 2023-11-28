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

extern int log2i(int v);

int main(int argc, char *argv[])
{
    int myrank, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc != 8)
    {
        if (!myrank) fprintf(stderr, "usage: %s <outprefix> <m> <n> <p> <r> <cond> <damp>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    char const *outprefix = argv[1];
    int m = atoi(argv[2]);
    int n = atoi(argv[3]);
    int p = atoi(argv[4]);
    int r = atoi(argv[5]);
    double cond = atof(argv[6]);
    double damp = atof(argv[7]);

    int s, q;
    double *A, *Aloc;

    q =  log2i(nprocs);

    generate_svd_dist_test(&A, &Aloc, &s, m, n, r, cond, damp, 0, MPI_COMM_WORLD);

    char fname[1024];
    snprintf(fname, 1024, "Aloc_%d.mtx", myrank);
    mmwrite(fname, Aloc, m, s);

    if (myrank == 0)
    {
        mmwrite("A.mtx", A, m, n);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
