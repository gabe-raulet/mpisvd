#ifndef SVD_SERIAL_H_
#define SVD_SERIAL_H_

/*
 * Computes an approximate p-truncated SVD A = Up*Sp*Vtp.
 */
int svd_serial
(
    double const *A, /* input m-by-n matrx */
    double **Up, /* output m-by-p matrix */
    double **Sp, /* output p-by-p diagonal matrix */
    double **Vtp, /* output p-by-n matrix */
    int m, /* rows of A */
    int n, /* columns of A */
    int p, /* rank approximation */
    int b /* number of seed nodes in binary topology */
);

#endif
