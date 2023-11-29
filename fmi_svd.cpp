#include <string>
#include <cstdio>
#include <fmi.h>
#include <svd_fmi.h>
#include <mmio_dense.h>
#include <iostream>

int main(int argc, char *argv[])
{
    int myrank = atoi(argv[1]);
    int nprocs = atoi(argv[2]);

    if (argc != 10)
    {
        if (!myrank) fprintf(stderr, "usage: %s <myrank> <nprocs> <outprefix> <m> <n> <p> <r> <cond> <damp>\n", argv[0]);
        return 1;
    }

    char const *outprefix = argv[3];
    int m = atoi(argv[4]);
    int n = atoi(argv[5]);
    int p = atoi(argv[6]);
    int r = atoi(argv[7]);
    double cond = atof(argv[8]);
    double damp = atof(argv[9]);

    std::string comm_name = std::to_string(std::time(nullptr));
    std::string config_path = "fmi/config/fmi_test.json";
    auto comm = FMI::Communicator(myrank, nprocs, config_path, comm_name);
    comm.hint(FMI::Utils::Hint::fast);
    comm.barrier();

    int s;
    double *A, *Aloc;
    if (generate_svd_fmi_test(&A, &Aloc, &s, m, n, r, cond, damp, 0, myrank, nprocs, comm) != 0)
    {
        if (!myrank) fprintf(stderr, "[error] generate_svd_fmi_test fails\n");
        return 1;
    }

    char fname[1024];

    if (myrank == 0)
    {
        snprintf(fname, 1024, "A_%s.mtx", outprefix);
        mmwrite(fname, A, m, n);
        free(A);
    }

    double *Up, *Sp, *Vtp;

    if (svd_fmi(Aloc, &Up, &Sp, &Vtp, m, n, p, 0, myrank, nprocs, comm) != 0)
    {
        return 1;
    }

    if (!myrank)
    {
        snprintf(fname, 1024, "Up_%s.mtx", outprefix);
        mmwrite(fname, Up, m, p);

        snprintf(fname, 1024, "Vtp_%s.mtx", outprefix);
        mmwrite(fname, Vtp, p, n);

        snprintf(fname, 1024, "Sp_%s.diag", outprefix);
        write_diag(fname, Sp, p);

        free(Up);
        free(Sp);
        free(Vtp);
    }

    return 0;
}