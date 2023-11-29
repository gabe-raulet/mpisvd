#ifndef SVD_SERIAL_H_
#define SVD_SERIAL_H_

int log2i(int v);

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

int generate_svd_test(double **A_ref, int m, int n, int r, double cond, double damp);

#endif
