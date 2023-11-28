#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "cblas.h"
#include "lapacke.h"
#include "svd_routines.h"
#include "linalg_routines.h"
#include "mmio_dense.h"

int svds_naive(const double *A, double *Up, double *Sp, double *Vpt, int m, int n, int p)
{
    assert(A != NULL && Up != NULL && Sp != NULL && Vpt != NULL && m >= n && n >= p && p >= 1);

    double *S, *U, *Vt;

    serial_thin_svd_lapack((double *)A, &S, &U, &Vt, m, n);

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

    double *W = malloc((2*d)*p*sizeof(double));

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
    double *tau = malloc(p*sizeof(double));


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

    double *Sp = malloc(p*sizeof(double));

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


