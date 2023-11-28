#ifndef SVD_ROUTINES_H_
#define SVD_ROUTINES_H_

int svds_naive(const double *A, double *Up, double *Sp, double *Vpt, int m, int n, int p);
int seed_node(double const *Ai, double *A1i, double *Vt1i, int m, int n, int q, int p);
int combine_node(double *Ak_2i_0, double *Vtk_2i_0, double *Ak_2i_1, double *Vtk_2i_1, double *Ak1_lj, double *Vtk1_lj, int m, int n, int k, int q, int p);

#endif
