#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include "kiss.h"
#include "cblas.h"
#include "lapacke.h"
#include "mmio_dense.h"
#include "svd_routines.h"

int myrank;
int nprocs;

typedef struct { int p, seed; char *label; } params_t;

int iseed[4];
int iseed_init();
int log2i(int v);

int svdalg_param_check(int m, int n, int p, int b);
int params_init(params_t *ps);
int parse_params(int argc, char *argv[], params_t *ps);
int usage(char *argv[], const params_t *ps);
double l2dist(const double *x, const double *y, int n);
int compute_errors(const double *A, const double *U, const double *Up, const double *S, const double *Sp, const double *Vt, const double *Vtp, int m, int n, int p, double errs[4], int fast);
int svd_dist(const double *Aloc, double *Up, double *Sp, double *Vtp, int m, int n, int p);
double get_clock_telapsed(struct timespec start, struct timespec end);

int main(int argc, char *argv[])
{
    params_t params;
    double gentime, svdtime, errtime;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (parse_params(argc, argv, &params) != 0)
    {
        MPI_Finalize();
        return 1;
    }

    if (params.seed <= 0)
        kiss_init();
    else
        kiss_seed((uint32_t)params.seed);

    iseed_init();

    int p=params.p, m, n;
    double *A, *Aloc, *S, *U, *Vt, *Scheck;


    if (!myrank)
    {
        FILE *f;
        char fname[1024];
        int tmp[2];

        double readtime = MPI_Wtime();

        snprintf(fname, 1024, "A_%s.mtx", params.label);
        A = mmread(fname, &m, &n);

        snprintf(fname, 1024, "U_%s.mtx", params.label);
        U = mmread(fname, tmp, tmp+1);
        assert(tmp[0] == m && tmp[1] == n);

        snprintf(fname, 1024, "Vt_%s.mtx", params.label);
        Vt = mmread(fname, tmp, tmp+1);
        assert(tmp[0] == n && tmp[1] == n);

        S = malloc(n*sizeof(double));

        snprintf(fname, 1024, "S_%s.diag", params.label);
        f = fopen(fname, "r");
        for (int i = 0; i < n; ++i) fscanf(f, "%lf\n", S+i);
        fclose(f);

        readtime = MPI_Wtime() - readtime;

        int show = n < 5? n : 5;
        fprintf(stderr, "[main[%d/%d]:read_truth::%.5f(s)] diag(S)[0..%d] = ", myrank+1, nprocs, readtime, show-1);
        for (int i = 0; i < show; ++i) fprintf(stderr, "%.3e,", S[i]);
        fprintf(stderr, "%s\n", n < 5? "" : "...");

        if (svdalg_param_check(m, n, p, nprocs))
        {
            MPI_Finalize();
            return 1;
        }
    }

    //MPI_Barrier(MPI_COMM_WORLD);

    //double *Up, *Sp, *Vtp;

    //svdtime = MPI_Wtime();

    //if (!myrank)
    //{
    //    Up = malloc(m*p*sizeof(double));
    //    Sp = malloc(p*sizeof(double));
    //    Vtp = malloc(p*n*sizeof(double));
    //}

    //int nloc = n / nprocs;

    //Aloc = malloc(m*nloc*sizeof(double));
    //MPI_Scatter(A, m*nloc, MPI_DOUBLE, Aloc, m*nloc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //if (svd_dist(Aloc, Up, Sp, Vtp, m, n, p) != 0)
    //{
    //    MPI_Finalize();
    //    return 1;
    //}

    //svdtime = MPI_Wtime() - svdtime;

    //if (!myrank) fprintf(stderr, "[main:svd_dist::%.5f(s)]\n", svdtime);

    //if (!myrank)
    //{
    //    double errs[4];

    //    errtime = MPI_Wtime();

    //    compute_errors(A, U, Up, S, Sp, Vt, Vtp, m, n, p, errs, fast);

    //    errtime = MPI_Wtime() - errtime;

    //    fprintf(stderr, "[main:compute_errors::%.5f(s)]\n", errtime);
    //    fprintf(stderr, "[main:compute_errors::*] err=%.18e (DLATMS-vs-RANDSVD) [A :: test matrix]\n", errs[0]);
    //    fprintf(stderr, "[main:compute_errors::*] err=%.18e (DLATMS-vs-RANDSVD) [S :: singular values]\n", errs[1]);

    //    if (!fast)
    //    {
    //        fprintf(stderr, "[main:compute_errors::*] err=%.18e (DLATMS-vs-RANDSVD) [U :: singular vectors (left)]\n", errs[2]);
    //        fprintf(stderr, "[main:compute_errors::*] err=%.18e (DLATMS-vs-RANDSVD) [V :: singular vectors (right)]\n", errs[3]);
    //    }

    //    free(A);
    //    free(S);
    //    free(Sp);
    //    free(U);
    //    free(Up);
    //    free(Vt);
    //    free(Vtp);
    //}

    //free(Aloc);

    MPI_Finalize();
    return 0;
}

int compute_errors(const double *A,
                   const double *U, const double *Up,
                   const double *S, const double *Sp,
                   const double *Vt, const double *Vtp,
                   int m, int n, int p,
                   double errs[4], int fast)
{
    double Aerr, Serr, Uerr, Verr;
    double *mem = malloc(m*m*sizeof(double));

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
        {
            double acc = 0;

            for (int k = 0; k < p; ++k)
            {
                acc += Up[i + k*m]*Sp[k]*Vtp[k + j*p];
            }

            mem[i + j*m] = acc - A[i + j*m];
        }

    Aerr = LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', m, n, mem, m);

    Serr = l2dist(S, Sp, p);

    if (!fast)
    {
        /*M = U@U.T - Up@Up.T m-by-m */

        for (int j = 0; j < m; ++j)
            for (int i = 0; i < m; ++i)
            {
                double acc = 0;

                for (int k = 0; k < p; ++k)
                {
                    acc += U[i + k*m]*U[j + k*m];
                    acc -= Up[i + k*m]*Up[j + k*m];
                }

                mem[i + j*m] = acc;
            }

        Uerr = LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', m, m, mem, m);

        /*M = Vt.T@Vt - Vtp.T@Vtp n-by-n */

        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
            {
                double acc = 0;

                for (int k = 0; k < p; ++k)
                {
                    acc += Vt[k + i*n]*Vt[k + j*n];
                    acc -= Vtp[k + i*p]*Vtp[k + j*p];
                }

                mem[i + j*n] = acc;
            }

        Verr = LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', n, n, mem, n);

        errs[2] = Uerr, errs[3] = Verr;
    }

    free(mem);

    errs[0] = Aerr, errs[1] = Serr;

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
    iseed[3] &= (iseed[3]^1); /* iseed[3] must be odd */
    return 0;
}

double l2dist(const double *x, const double *y, int n)
{
    double v = 0;

    for (int i = 0; i < n; ++i)
    {
        v += ((x[i] - y[i]) * (x[i] - y[i]));
    }

    return sqrt(v);
}

int params_init(params_t *ps)
{
    ps->p = 10;
    ps->seed = -1;
    ps->label = NULL;

    return 0;
}

int usage(char *argv[], const params_t *ps)
{
    if (!myrank)
    {
        fprintf(stderr, "Usage: %s [options]\n", argv[0]);
        fprintf(stderr, "Options: -p INT    SVD truncation parameter [%d]\n", ps->p);
        fprintf(stderr, "         -s INT    RNG seed [%d]\n", ps->seed);
        fprintf(stderr, "         -i STR    input label\n");
        fprintf(stderr, "         -h        help message\n");
    }
    return 1;
}

int parse_params(int argc, char *argv[], params_t *ps)
{
    int c;
    params_init(ps);

    while ((c = getopt(argc, argv, "p:s:i:h")) >= 0)
    {
        if      (c == 'p') ps->p = atoi(optarg);
        else if (c == 's') ps->seed = atoi(optarg);
        else if (c == 'i') ps->label = optarg;
        else if (c == 'h')
        {
            params_init(ps);
            return usage(argv, ps);
        }
    }

    if (ps->label == NULL)
    {
        if (!myrank) fprintf(stderr, "error: missing -i parameter\n");
        return 1;
    }

    return 0;
}

int svdalg_param_check(int m, int n, int p, int b)
{
    if (!(1 <= p && p <= n && n <= m) || (m&(m-1)) || (n&(n-1)))
    {
        if (!myrank) fprintf(stderr, "[error::svdalg_param_check][m=%d,n=%d,p=%d] must have 1 <= p <= n <= m with n and m being powers of 2\n", m, n, p);
        return 1;
    }

    if (b <= 1 || (b&(b-1)) || n % b != 0)
    {
        if (!myrank) fprintf(stderr, "[error:svdalg_param_check][b=%d,n=%d] must have b > 1 and n %% b == 0 with b being a power of 2\n", b, n);
        return 1;
    }

    return 0;
}

double get_clock_telapsed(struct timespec start, struct timespec end)
{
    uint64_t diff = 1000000000L * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec);
    return (diff / 1000000000.0);
}

int svd_dist(const double *Aloc, double *Up, double *Sp, double *Vtp, int m, int n, int p)
{
    int myrank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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
        MPI_Send(A1i, m*p, MPI_DOUBLE, myrank-1, myrank, MPI_COMM_WORLD);
        MPI_Send(Vt1i, p*s, MPI_DOUBLE, myrank-1, myrank+nprocs, MPI_COMM_WORLD);
    }
    else
    {
        memcpy(Amem, A1i, m*p*sizeof(double));
        memcpy(Vtmem, Vt1i, p*s*sizeof(double));
        MPI_Recv(&Amem[m*p], m*p, MPI_DOUBLE, myrank+1, myrank+1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&Vtmem[p*s], p*s, MPI_DOUBLE, myrank+1, myrank+1+nprocs, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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

            MPI_Send(Ak1_lj, m*p, MPI_DOUBLE, dest, Atag, MPI_COMM_WORLD);
            MPI_Send(Vtk1_lj, p*2*d, MPI_DOUBLE, dest, Vtag, MPI_COMM_WORLD);
        }
        else if ((myrank % (1 << (k+1))) == 0)
        {

            int source = myrank + (1 << k);
            int Atag = myrank + (1 << k);
            int Vtag = myrank + (1 << k) + nprocs;

            MPI_Recv(&Amem[m*p], m*p, MPI_DOUBLE, source, Atag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&Vtmem[p*2*d], p*2*d, MPI_DOUBLE, source, Vtag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            /*printf("k=%d: rank %d received %d doubles from rank %d with tag %d\n", k, myrank, m*p, source, Atag);*/
            /*printf("k=%d: rank %d received %d doubles from rank %d with tag %d\n", k, myrank, p*2*d, source, Vtag);*/
        }

    }

    MPI_Barrier(MPI_COMM_WORLD);

    double *Aq1_11, *Aq1_12, *Vtq1_11, *Vtq1_12;

    if (!myrank)
    {
        Aq1_11 = &Amem[0];
        Aq1_12 = &Amem[m*p];
        Vtq1_11 = &Vtmem[0];
        Vtq1_12 = &Vtmem[(n*p)>>1];

        extract_node(Aq1_11, Vtq1_11, Aq1_12, Vtq1_12, Up, Sp, Vtp, m, n, q, p);
    }

    free(A1i);
    free(Vt1i);

    MPI_Barrier(MPI_COMM_WORLD);
    return 0;
}
