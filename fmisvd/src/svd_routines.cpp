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

    work = (double *) malloc(5*drank*sizeof(double));
    S = (double *) malloc(drank*sizeof(double));
    U = (double *) malloc(m*drank*sizeof(double));
    Vt = (double *) malloc(drank*n*sizeof(double));

    LAPACKE_dgesvd
    (
        LAPACK_COL_MAJOR,
       'S',
       'S',
        m,
        n,
        Acast,
        m,
        S,
        U,
        m,
        Vt,
        drank,
        work
    );

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

    double *Aki = (double *) malloc(m*(2*p)*sizeof(double));

    memcpy(&Aki[0],   Ak_2i_0, m*p*sizeof(double));
    memcpy(&Aki[m*p], Ak_2i_1, m*p*sizeof(double));

    double *Vhtki = (double *) calloc((2*p)*(2*d), sizeof(double));

    for (int j = 0; j < d; ++j)
    {
        memcpy(&Vhtki[j*(2*p)], &Vtk_2i_0[j*p], p*sizeof(double));
        memcpy(&Vhtki[(j+d)*(2*p)+p], &Vtk_2i_1[j*p], p*sizeof(double));
    }

    double *Uki = (double *) malloc(m*p*sizeof(double));
    double *Ski = (double *) malloc(p*sizeof(double));
    double *Vtki = (double *) malloc(p*(2*p)*sizeof(double));

    svds_naive(Aki, Uki, Ski, Vtki, m, 2*p, p);

    /*
     * Compute USki = Uki*Ski.
     */

    double *USki = Uki;

    for (int j = 0; j < p; ++j)
        for (int i = 0; i < m; ++i)
            USki[i + j*m] *= Ski[j];

    free(Ski);

    /*
     * Compute W = Tr(Vhtki)*Tr(Vtki).
     *
     * W :: 2d-by-p
     * Tr(Vhtki) :: 2d-by-2p
     * Tr(Vtki) :: 2p-by-p
     */

    double *W = (double *) malloc((2*d)*p*sizeof(double));

    cblas_dgemm
    (
        CblasColMajor, /* all matrices stored column-major */
           CblasTrans, /* transpose Vhtki */
           CblasTrans, /* transpose Vtki */
                  2*d, /* number of rows of W (and Tr(Vhtki)) */
                    p, /* number of columns of W (and Tr(Vtki)) */
                  2*p, /* number of columns of Tr(Vhtki) (and number of rows of Tr(Vtki)) */
                  1.0, /* the alpha in "W <- alpha*Tr(Vhtki)*Tr(Vtki) + beta*W" */
                Vhtki, /* Vhtki matrix */
                  2*p, /* leading dimension of Vhtki */
                 Vtki, /* Vtki matrix */
                    p, /* leading dimension of Vtki */
                  0.0, /* the beta in "W <- alpha*Tr(Vhtki)*Tr(Vtki) + beta*W" */
                    W, /* W matrix */
                  2*d  /* leading dimension of W */
    );

    free(Vtki);
    free(Vhtki);

    /*
     * Compute QR-factorization W = Qki*Rki.
     *
     * W :: 2d-by-p
     * Qki :: 2d-by-p
     * Rki :: p-by-p
     */

    assert(2*d >= p);
    double *tau = (double *) malloc(p*sizeof(double));


    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, 2*d, p, W, 2*d, tau);

    /*
     * W now contains Rki in its leading p-by-p upper triangular portion.
     * Compute Rki^{-1} in-place with DTRTRI. On exit, The upper triangular
     * portion will then store Rki^{-1}, and lucky for us the entries in the
     * strictly lower triangular portion of W are untouched, so we can use the
     * reflectors stored there to reconstruct Q after with DORGQR.
     */

    LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'U', 'N', p, W, 2*d);

    /*
     * Compute Aki = USki * inv(Rki).
     *
     * Aki :: m-by-p
     * USki :: m-by-p
     * inv(Rki) :: p-by-p :: upper triangular stored in in W
     */

    cblas_dtrmm
    (
        CblasColMajor, /* all matrices stored column-major */
           CblasRight, /* triangular matrix on the right */
           CblasUpper, /* Rki is upper triangular */
         CblasNoTrans, /* don't transpose Rki */
         CblasNonUnit, /* Rki not necessarily unit triangular */
                    m, /* number of rows of USki */
                    p, /* number of columns of USki */
                  1.0, /* the alpha in Aki := alpha*USki*inv(Rki) */
                    W, /* inv(Rki) contained in the upper triangular portion of W */
                  2*d, /* leading dimension of W */
                 USki, /* USki on entry, Aki on exit */
                    m  /* leading dimension of USki */
    );

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

    double *Sp = (double *) malloc(p*sizeof(double));

    svds_naive(Ai, A1i, Sp, Vt1i, m, s, p);

    /*
     * Compute Up*Sp and write it to A1i.
     */

    for (int j = 0; j < p; ++j)
        for (int i = 0; i < m; ++i)
            A1i[i + m*j] *= Sp[j];

    free(Sp);
    return 0;
}

int extract_node(double *Aq1_11, double *Vtq1_11, double *Aq1_12, double *Vtq1_12, double **U, double **S, double **Vt, int m, int n, int q, int p)
{
    int b = 1 << q;
    int s = n / b;
    int d = s * (1 << (q-1));

    assert(2*d == n);

    double *Aq1 = (double *) malloc(m*(2*p)*sizeof(double));

    memcpy(&Aq1[0],   Aq1_11, m*p*sizeof(double));
    memcpy(&Aq1[m*p], Aq1_12, m*p*sizeof(double));

    double *Vhtq1 = (double *) calloc((2*p)*(2*d), sizeof(double));

    for (int j = 0; j < d; ++j)
    {
        memcpy(&Vhtq1[j*(2*p)], &Vtq1_11[j*p], p*sizeof(double));
        memcpy(&Vhtq1[(j+d)*(2*p)+p], &Vtq1_12[j*p], p*sizeof(double));
    }

    double *Uq = (double *) malloc(m*p*sizeof(double));
    double *Sq = (double *) malloc(p*sizeof(double));
    double *Vtq = (double *) malloc(p*(2*p)*sizeof(double));

    svds_naive(Aq1, Uq, Sq, Vtq, m, 2*p, p);

    double *USq = Uq;

    for (int j = 0; j < p; ++j)
        for (int i = 0; i < m; ++i)
            USq[i + j*m] *= Sq[j];

    free(Sq);

    double *W = (double *) malloc(n*p*sizeof(double));

    cblas_dgemm
    (
        CblasColMajor, /* all matrices stored column-major */
           CblasTrans, /* transpose Vhtq1 */
           CblasTrans, /* transpose Vtq */
                    n, /* number of rows of W (and Tr(Vhtq1)) */
                    p, /* number of columns of W (and Tr(Vtq)) */
                  2*p, /* number of columns of Tr(Vhtq1) (and number of rows of Tr(Vtq)) */
                  1.0, /* the alpha in "W <- alpha*Tr(Vhtq1)*Tr(Vtq) + beta*W" */
                Vhtq1, /* Vhtq1 matrix */
                  2*p, /* leading dimension of Vhtq1 */
                  Vtq, /* Vtq matrix */
                    p, /* leading dimension of Vtq */
                  0.0, /* the beta in "W <- alpha*Tr(Vhtq1)*Tr(Vtq) + beta*W" */
                    W, /* W matrix */
                    n  /* leading dimension of W */
    );

    free(Vtq);
    free(Vhtq1);

    assert(n >= p);
    double *tau = (double *) malloc(p*sizeof(double));

    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, n, p, W, n, tau);
    LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'U', 'N', p, W, n);

    cblas_dtrmm
    (
        CblasColMajor, /* all matrices stored column-major */
           CblasRight, /* triangular matrix on the right */
           CblasUpper, /* Rq is upper triangular */
         CblasNoTrans, /* don't transpose Rq */
         CblasNonUnit, /* Rq not necessarily unit triangular */
                    m, /* number of rows of USq */
                    p, /* number of columns of USq */
                  1.0, /* the alpha in Aq := alpha*USq*inv(Rq) */
                    W, /* inv(Rq) contained in the upper triangular portion of W */
                  2*d, /* leading dimension of W */
                  USq, /* USq on entry, Aq on exit */
                    m  /* leading dimension of USq */
    );

    LAPACKE_dorgqr(LAPACK_COL_MAJOR, n, p, p, W, n, tau);

    double *Aq = USq;
    double *Qq = W;
    double *Uc = (double *) malloc(m*p*sizeof(double));
    double *Sc = (double *) malloc(p*sizeof(double));
    double *Vtp = (double *) malloc(p*n*sizeof(double));
    double *Vtc = (double *) malloc(p*n*sizeof(double));

    svds_naive(Aq, Uc, Sc, Vtp, m, p, p);

    /*
     * Uc is m-by-p
     * Sc is p-by-p
     * Vtp is p-by-p
     * Qq is n-by-p
     *
     * Vtc = Vtp*Qq.T is p-by-n
     */

    cblas_dgemm
    (
        CblasColMajor, /* all matrices stored column-major */
         CblasNoTrans, /* don't transpose A */
           CblasTrans, /* transpose B */
                    p, /* number of rows of C */
                    n, /* number of columns of C */
                    p, /* number of columns of A */
                  1.0, /* the alpha in "C <- alpha*op(A)*op(B) + beta*C" */
                  Vtp, /* A matrix */
                    p, /* leading dimension of A */
                   Qq, /* B matrix */
                    n, /* leading dimension of B */
                  0.0, /* beta */
                  Vtc, /* C matrix */
                    p  /* leading dimension of C */
    );

    *U = Uc;
    *S = Sc;
    *Vt = Vtc;

    return 0;
}


