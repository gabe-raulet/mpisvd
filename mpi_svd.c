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

int main(int argc, char *argv[])
{
    int myrank, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc != 9)
    {
        if (!myrank) fprintf(stderr, "usage: %s <outprefix> <m> <n> <p> <q> <r> <cond> <damp>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    char const *outprefix = argv[1];
    int m = atoi(argv[2]);
    int n = atoi(argv[3]);
    int p = atoi(argv[4]);
    int q = atoi(argv[5]);
    int r = atoi(argv[6]);
    double cond = atof(argv[7]);
    double damp = atof(argv[8]);

    int s;
    double *A, *Aloc;

    generate_svd_dist_test(&A, &Aloc, &s, m, n, r, cond, damp, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}
