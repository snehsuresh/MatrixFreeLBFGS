#ifndef OBJECTIVE_H
#define OBJECTIVE_H

#include <stddef.h>

// For Rosenbrock, we define the objective and gradient.
// Note: n must be even.
void init_sparse_mask(size_t n, double sparsity);
double objective_function(const double *x, double *grad, size_t n);

#endif // OBJECTIVE_H
