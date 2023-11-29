#include "../inc/svd_fmi.h"
#include "../inc/fmi_wrapper.h"
#include "../inc/kiss.h"
#include "../inc/svd_routines.h"
#include "../inc/svd_serial.h"
#include "lapacke.h"
#include <fmi.h>
#include <iostream>
#include <cstring>

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
)
{
    double *Aroot, *Apart;

    if (myrank == root)
    {
        if (generate_svd_test(&Aroot, m, n, r, cond, damp) != 0)
            return 1;
    }

    assert(n % nprocs == 0);

    int nloc = n / nprocs;

    Apart = (double*) malloc(m*nloc*sizeof(double));

    // Different from MPI
    fmi_scatter((void *) Aroot, m * n * sizeof(double), (void *) Apart, m * nloc * sizeof(double), root, comm);

    *s = nloc;
    *Aloc = Apart;
    *A = (myrank == root)? Aroot : NULL;

    comm.barrier();
    return 0;
}

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
)
{
    assert(n % nprocs == 0);

    int s = n / nprocs;

    double *A1i, *Vt1i;

    A1i = (double *) malloc(m*s*sizeof(double));
    Vt1i = (double *) malloc(s*p*sizeof(double));

    int q = log2i(nprocs);

    seed_node(Aloc, A1i, Vt1i, m, n, q, p);

    double *Amem = (double *) malloc(2*m*p*sizeof(double));
    double *Vtmem = (double *) malloc(n*p*sizeof(double)); /* this should be allocated with less memory depending on what myrank is */

    if (myrank % 2 != 0)
    {
        fmi_send((void *) A1i, m*p*sizeof(double), myrank-1, comm);
        fmi_send((void *) Vt1i, p*s* sizeof(double), myrank-1, comm);
    }
    else
    {
        std::memcpy(Amem, A1i, m*p*sizeof(double));
        std::memcpy(Vtmem, Vt1i, p*s*sizeof(double));
        fmi_recv((void *) (&Amem[m*p]), m*p*sizeof(double), myrank+1, comm);
        fmi_recv((void *) (&Vtmem[p*s]), p*s*sizeof(double), myrank+1, comm);
    }

    double *Ak_2i_0, *Vtk_2i_0, *Ak_2i_1, *Vtk_2i_1, *Ak1_lj, *Vtk1_lj;

    for (int k = 1; k < q; ++k)
    {
        int c = 1 << (q-k); /* nodes on this level */
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

            fmi_send((void *) Ak1_lj, m*p*sizeof(double), dest, comm);
            fmi_send((void *) Vtk1_lj, p*2*d*sizeof(double), dest, comm);
        }
        else if ((myrank % (1 << (k+1))) == 0)
        {

            int source = myrank + (1 << k);
            int Atag = myrank + (1 << k);
            int Vtag = myrank + (1 << k) + nprocs;

            fmi_recv((void *) (&Amem[m*p]), m*p*sizeof(double), source, comm);
            fmi_recv((void *) (&Vtmem[p*2*d]), p*2*d*sizeof(double), source, comm);

            /*printf("k=%d: rank %d received %d doubles from rank %d with tag %d\n", k, myrank, m*p, source, Atag);*/
            /*printf("k=%d: rank %d received %d doubles from rank %d with tag %d\n", k, myrank, p*2*d, source, Vtag);*/
        }

    }

    comm.barrier();

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