#ifndef LINALG_ROUTINES_H_
#define LINALG_ROUTINES_H_

int serial_thin_svd_lapack(double *A, double **SIGMA_ref, double **U_ref, double **Vt_ref, int m, int n);
int serial_qr_lapack(double *A, double **Q_ref, double **R_ref, int m, int n);

#endif
