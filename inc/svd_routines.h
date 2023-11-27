#ifndef SVD_ROUTINES_H_
#define SVD_ROUTINES_H_

int svds_naive(const double *A, double *Up, double *Sp, double *Vpt, int m, int n, int p);
int seed_node(double const *A1i, double *Aki, double *Vtki, int m, int s, int p);
int combine_node(double *Aki12, double *Vtki12, int l, int m, int s, int p);

#endif
