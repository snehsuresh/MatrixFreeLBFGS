#ifndef OBJECTIVE_H
#define OBJECTIVE_H

#include <stddef.h>

// For the Rosenbrock function, which is defined in pairs.
// n must be even.
void init_sparse_mask(size_t n, double sparsity);
double objective_function(const double *x, double *grad, size_t n);

#endif // OBJECTIVE_H
