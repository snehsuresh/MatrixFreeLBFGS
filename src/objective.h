#ifndef OBJECTIVE_H
#define OBJECTIVE_H

#include <stddef.h>

// Evaluate the objective function and its gradient at x.
// The objective function simulates a sparse quadratic function:
// f(x) = 0.5 * sum_{i in S} (x[i] - 1)^2, where S is a subset of indices.
// The gradient is computed only for indices in S, simulating sparsity.
double objective_function(const double *x, double *grad, size_t n);

// Initialize the sparse mask. Must be called before using objective_function.
void init_sparse_mask(size_t n, double sparsity); // sparsity: fraction of indices that are "active"

#endif // OBJECTIVE_H
