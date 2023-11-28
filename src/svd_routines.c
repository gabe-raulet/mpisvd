#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "cblas.h"
#include "lapacke.h"
#include "svd_routines.h"
#include "linalg_routines.h"

int naive_transpose(double *At, const double *A, int m, int n)
{
    /*
     * At := Tr(A) where A is m-by-n. Hence At is n-by-m.
     */

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            At[j + i*n] = A[i + j*m];

    return 0;
}

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

int combine_node(double *Aki12, double *Vtki12, int l, int m, int s, int p)
{
    int u = (1 << (l-1))*s;

    double *Aki;    /* m-by-2p @ &Aki12[0] */
    double *Vtki1;  /* p-by-u @ &Vtki12[0] */
    double *Vtki2;  /* p-by-u @ &Vtki12[p*u] */
    double *Vhtki;  /* 2p-by-2u */
    double *Uki;    /* m-by-p */
    double *Ski;    /* p-by-p (diagonal) */
    double *USki;   /* m-by-p */
    double *Vtki;   /* p-by-2p */
    double *W;      /* 2u-by-p :: == Tr(Vhtki)*Tr(Vtki) */
    double *Qki;    /* 2u-by-p */
    double *Rki;    /* p-by-p */
    double *tau;

    (void)Qki;
    (void)Rki;

    Aki = Aki12;
    Vtki1 = Vtki12;
    Vtki2 = &Vtki12[p*u];

    /*
     * Take the p-truncated SVD of Aki = Uki*Ski*Vtki.
     */

    Uki = malloc(m*p*sizeof(double));
    Ski = malloc(p*sizeof(double));
    Vtki = malloc(p*(2*p)*sizeof(double));

    svds_naive(Aki, Uki, Ski, Vtki, m, 2*p, p);

    /*
     * Compute USki = Uki*Ski.
     */

    USki = Uki;

    for (int j = 0; j < p; ++j)
        for (int i = 0; i < m; ++i)
            USki[i + j*m] *= Ski[j];

    free(Ski);

    /*
     * Construct Vhtki = [Vtki1 0; 0 Vtki2].
     */

    Vhtki = calloc((2*p)*(2*u), sizeof(double));

    for (int j = 0; j < u; ++j)
    {
        memcpy(&Vhtki[j*(2*p)], &Vtki1[j*p], p*sizeof(double));
        memcpy(&Vhtki[(j+u)*(2*p)+p], &Vtki2[j*p], p*sizeof(double));
    }

    /*
     * Compute W = Tr(Vhtki)*Tr(Vtki).
     *
     * W :: 2u-by-p
     * Tr(Vhtki) :: 2u-by-2p
     * Tr(Vtki) :: 2p-by-p
     */

    W = malloc((2*u)*p*sizeof(double));

    cblas_dgemm
    (
        CblasColMajor, /* all matrices stored column-major */
           CblasTrans, /* transpose Vhtki */
           CblasTrans, /* transpose Vtki */
                  2*u, /* number of rows of W (and Tr(Vhtki)) */
                    p, /* number of columns of W (and Tr(Vtki)) */
                  2*p, /* number of columns of Tr(Vhtki) (and number of rows of Tr(Vtki)) */
                  1.0, /* the alpha in "W <- alpha*Tr(Vhtki)*Tr(Vtki) + beta*W" */
                Vhtki, /* Vhtki matrix */
                  2*p, /* leading dimension of Vhtki */
                 Vtki, /* Vtki matrix */
                    p, /* leading dimension of Vtki */
                  0.0, /* the beta in "W <- alpha*Tr(Vhtki)*Tr(Vtki) + beta*W" */
                    W, /* W matrix */
                  2*u  /* leading dimension of W */
    );

    free(Vtki);
    free(Vhtki);

    /*
     * Compute QR-factorization W = Qki*Rki.
     *
     * W :: 2u-by-p
     * Qki :: 2u-by-p
     * Rki :: p-by-p
     */

    assert(2*u >= p);

    tau = malloc(p*sizeof(double));

    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, 2*u, p, W, 2*u, tau);

    /*
     * W now contains Rki in its leading p-by-p upper triangular portion.
     * Compute Rki^{-1} in-place with DTRTRI. On exit, The upper triangular
     * portion will then store Rki^{-1}, and lucky for us the entries in the
     * strictly lower triangular portion of W are untouched, so we can use the
     * reflectors stored there to reconstruct Q after with DORGQR.
     */

    LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'U', 'N', p, W, 2*u);

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
                    m, /* leading dimension of W */
                 USki, /* USki on entry, Aki on exit */
                    m  /* leading dimension of USki */
    );

    LAPACKE_dorgqr(LAPACK_COL_MAJOR, 2*u, p, p, W, 2*u, tau);
    free(tau);

    memcpy(Aki, USki, m*p*sizeof(double));

    naive_transpose(Vtki12, W, 2*u, p);

    free(Uki);
    free(W);
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


