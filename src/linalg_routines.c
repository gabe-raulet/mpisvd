
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "linalg_routines.h"
#include "lapacke.h"

#define min(a, b) (((a) < (b))? (a) : (b))
#define max(a, b) (((a) > (b))? (a) : (b))

int serial_thin_svd_lapack(double *A, double **SIGMA_ref, double **U_ref, double **Vt_ref, int m, int n)
{
    assert(A != NULL && SIGMA_ref != NULL && m >= 1 && n >= 1);

    int drank = min(m, n); /* minimum dimension */
    int lda = m; /* leading dimension of array A */
    int ldu = U_ref? m : 1; /* leading dimension of array U (1 if not calculated) */
    int ldvt = Vt_ref? drank : 1; /* leading dimension of array Vt (1 if not calculated) */
    int lwork = 5 * drank; /* workspace array length */
    double *work = malloc(lwork*sizeof(double)); /* workspace */
    double *SIGMA = malloc(drank*sizeof(double)); /* singular values array */
    double *U = U_ref? malloc(ldu*drank*sizeof(double)) : NULL; /* left singular vectors array */
    double *Vt = Vt_ref? malloc(ldvt*n*sizeof(double)) : NULL; /* right singular vectors array */
    char jobu = U_ref? 'S' : 'N';
    char jobvt = Vt_ref? 'S' : 'N';

    assert(work != NULL);
    assert(SIGMA != NULL);
    if (U_ref) assert(U != NULL);
    if (Vt_ref) assert(Vt != NULL);

    int info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, jobu, jobvt, m, n, A, lda, SIGMA, U, ldu, Vt, ldvt, work);

    assert(info == 0);

    *SIGMA_ref = SIGMA;

    if (U_ref) *U_ref = U;
    if (Vt_ref) *Vt_ref = Vt;

    free(work);

    return 0;
}

int serial_qr_lapack(double *A, double **Q_ref, double **R_ref, int m, int n)
{
    assert (A != NULL && R_ref != NULL && m >= 1 && n >= 1 && m >= n);

    int drank = n; /* minimum dimension */
    int lda = m; /* leading dimension of A */
    double *tau = malloc(drank*sizeof(double));
    double *R = calloc(n*n, sizeof(double));
    assert(tau != NULL && R != NULL);

    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, A, lda, tau);

    for (int j = 0; j < n; ++j) /* faster to go column by column since it is column-major */
        for (int i = 0; i <= j; ++i) /* R is upper triangular */
            R[i + j*n] = A[i + j*m]; /* R(i,j) = A(i,j) */

    *R_ref = R;

    if (Q_ref)
    {
        double *Q = malloc(m*n*sizeof(double));
        assert(Q != NULL);

        LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, n, drank, A, lda, tau);

        memcpy(Q, A, m*n*sizeof(double));
        *Q_ref = Q;
    }

    free(tau);
    return 0;
}

