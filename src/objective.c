#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "objective.h"

// In this version, we use full gradients (no sparsity), so init_sparse_mask does nothing.
void init_sparse_mask(size_t n, double sparsity) {
    (void)n;
    (void)sparsity;
}

// Rosenbrock function and gradient.
// f(x) = sum_{i=1}^{n/2} [ (1 - x[2*i-1])^2 + 100*(x[2*i] - x[2*i-1]^2)^2 ]
double objective_function(const double *x, double *grad, size_t n) {
    if (n % 2 != 0) {
        fprintf(stderr, "[ERROR] Rosenbrock function requires an even dimension.\n");
        exit(EXIT_FAILURE);
    }
    double f = 0.0;
    // Zero out gradient.
    for (size_t i = 0; i < n; i++) {
        grad[i] = 0.0;
    }
    for (size_t i = 0; i < n/2; i++) {
        size_t idx1 = 2 * i;
        size_t idx2 = 2 * i + 1;
        double x1 = x[idx1];
        double x2 = x[idx2];
        double t1 = 1.0 - x1;
        double t2 = x2 - x1 * x1;
        f += t1 * t1 + 100.0 * t2 * t2;
        grad[idx1] = -2.0 * t1 - 400.0 * x1 * t2;
        grad[idx2] = 200.0 * t2;
    }
    return f;
}
