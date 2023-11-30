#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "kiss.h"
#include "cblas.h"
#include "lapacke.h"
#include "svd_routines.h"
#include "mmio_dense.h"

int svds_naive(const double *A, double *Up, double *Sp, double *Vpt, int m, int n, int p)
{
    assert(A != NULL && Up != NULL && Sp != NULL && Vpt != NULL && m >= n && n >= p && p >= 1);

    double *S, *U, *Vt, *work, *Acast = (double *)A;
    int drank = m < n? m : n;

    work = malloc(5*drank*sizeof(double));
    S = malloc(drank*sizeof(double));
    U = malloc(m*drank*sizeof(double));
    Vt = malloc(drank*n*sizeof(double));

    LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'S', m, n, Acast, m, S, U, m, Vt, drank, work);

    memcpy(Up, U, p*m*sizeof(double));
    memcpy(Sp, S, p*sizeof(double));

    double *Vt_ptr = Vt;
    double *Vpt_ptr = Vpt;

    for (int j = 0; j < n; ++j)
    {
        memcpy(Vpt_ptr, Vt_ptr, p*sizeof(double));

        Vpt_ptr += p;
        Vt_ptr += n;
    }

    free(S);
    free(U);
    free(Vt);

    return 0;
}

int combine_node(double *Ak_2i_0, double *Vtk_2i_0, double *Ak_2i_1, double *Vtk_2i_1, double *Ak1_lj, double *Vtk1_lj, int m, int n, int k, int q, int p)
{
    int b = 1 << q;
    int s = n / b;
    int d = (1 << (k-1)) * s;

    double *Aki = malloc(m*(2*p)*sizeof(double));

    memcpy(&Aki[0],   Ak_2i_0, m*p*sizeof(double));
    memcpy(&Aki[m*p], Ak_2i_1, m*p*sizeof(double));

    double *Vhtki = calloc((2*p)*(2*d), sizeof(double));

    for (int j = 0; j < d; ++j)
    {
        memcpy(&Vhtki[j*(2*p)], &Vtk_2i_0[j*p], p*sizeof(double));
        memcpy(&Vhtki[(j+d)*(2*p)+p], &Vtk_2i_1[j*p], p*sizeof(double));
    }

    double *Uki = malloc(m*p*sizeof(double));
    double *Ski = malloc(p*sizeof(double));
    double *Vtki = malloc(p*(2*p)*sizeof(double));

    svds_naive(Aki, Uki, Ski, Vtki, m, 2*p, p);

    double *USki = Uki;

    for (int j = 0; j < p; ++j)
        for (int i = 0; i < m; ++i)
            USki[i + j*m] *= Ski[j];

    free(Ski);

    double *W = malloc((2*d)*p*sizeof(double));

    cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, 2*d, p, 2*p, 1.0, Vhtki, 2*p, Vtki, p, 0.0, W, 2*d);

    free(Vtki);
    free(Vhtki);

    assert(2*d >= p);
    double *tau = malloc(p*sizeof(double));

    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, 2*d, p, W, 2*d, tau);
    LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'U', 'N', p, W, 2*d);

    cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, m, p, 1.0, W, 2*d, USki, m);

    LAPACKE_dorgqr(LAPACK_COL_MAJOR, 2*d, p, p, W, 2*d, tau);

    memcpy(Ak1_lj, USki, m*p*sizeof(double));

    /* Vtk1_lj[j,i] = W[i,j]; W is 2d-by-p */

    for (int j = 0; j < p; ++j)
        for (int i = 0; i < 2*d; ++i)
            Vtk1_lj[j + i*p] = W[i + j*2*d];

    free(Aki);
    free(Uki);
    free(W);
    free(tau);

    return 0;
}

int seed_node(double const *Ai, double *A1i, double *Vt1i, int m, int n, int q, int p)
{
    int b = 1 << q;
    int s = n / b;

    double *Sp = malloc(p*sizeof(double));

    svds_naive(Ai, A1i, Sp, Vt1i, m, s, p);

    for (int j = 0; j < p; ++j)
        for (int i = 0; i < m; ++i)
            A1i[i + m*j] *= Sp[j];

    free(Sp);
    return 0;
}

int extract_node(double *Aq1_11, double *Vtq1_11, double *Aq1_12, double *Vtq1_12, double *U, double *S, double *Vt, int m, int n, int q, int p)
{
    int b = 1 << q;
    int s = n / b;
    int d = s * (1 << (q-1));

    assert(2*d == n);

    double *Aq1 = malloc(m*(2*p)*sizeof(double));

    memcpy(&Aq1[0],   Aq1_11, m*p*sizeof(double));
    memcpy(&Aq1[m*p], Aq1_12, m*p*sizeof(double));

    double *Vhtq1 = calloc((2*p)*(2*d), sizeof(double));

    for (int j = 0; j < d; ++j)
    {
        memcpy(&Vhtq1[j*(2*p)], &Vtq1_11[j*p], p*sizeof(double));
        memcpy(&Vhtq1[(j+d)*(2*p)+p], &Vtq1_12[j*p], p*sizeof(double));
    }

    double *Uq = malloc(m*p*sizeof(double));
    double *Sq = malloc(p*sizeof(double));
    double *Vtq = malloc(p*(2*p)*sizeof(double));

    svds_naive(Aq1, Uq, Sq, Vtq, m, 2*p, p);

    double *USq = Uq;

    for (int j = 0; j < p; ++j)
        for (int i = 0; i < m; ++i)
            USq[i + j*m] *= Sq[j];

    free(Sq);

    double *W = malloc(n*p*sizeof(double));

    cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, n, p, 2*p, 1.0, Vhtq1, 2*p, Vtq, p, 0.0, W, n);

    free(Vtq);
    free(Vhtq1);

    assert(n >= p);
    double *tau = malloc(p*sizeof(double));

    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, n, p, W, n, tau);
    LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'U', 'N', p, W, n);

    cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, m, p, 1.0, W, 2*d, USq, m);

    LAPACKE_dorgqr(LAPACK_COL_MAJOR, n, p, p, W, n, tau);

    double *Aq = USq;
    double *Qq = W;
    double *Vtp = malloc(p*n*sizeof(double));

    svds_naive(Aq, U, S, Vtp, m, p, p);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, p, n, p, 1.0, Vtp, p, Qq, n, 0.0, Vt, p );

    return 0;
}


