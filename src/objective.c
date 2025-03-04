#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "objective.h"

static int *mask = NULL;
static size_t mask_size = 0;

// Initialize the sparse mask.
// 'sparsity' is the fraction of indices that are active (non-zero gradient).
void init_sparse_mask(size_t n, double sparsity) {
    if (mask) free(mask);
    mask = (int *)malloc(n * sizeof(int));
    mask_size = n;
    for (size_t i = 0; i < n; i++) {
        double r = (double)rand() / RAND_MAX;
        // If r < sparsity, mark as active (1), else inactive (0).
        mask[i] = (r < sparsity) ? 1 : 0;
    }
}

// Objective function: f(x) = 0.5 * sum_{i in S} (x[i] - 1)^2
// Also compute gradient: grad[i] = (x[i] - 1) if i in S, else 0.
double objective_function(const double *x, double *grad, size_t n) {
    double f = 0.0;
    // If mask is not initialized, assume full gradient.
    if (!mask) {
        for (size_t i = 0; i < n; i++) {
            double diff = x[i] - 1.0;
            f += 0.5 * diff * diff;
            grad[i] = diff;
        }
    } else {
        // Optionally parallelize gradient computation if OpenMP is available.
        #ifdef _OPENMP
        #pragma omp parallel for reduction(+:f)
        #endif
        for (size_t i = 0; i < n; i++) {
            if (mask[i]) {
                double diff = x[i] - 1.0;
                f += 0.5 * diff * diff;
                grad[i] = diff;
            } else {
                grad[i] = 0.0;
            }
        }
    }
    return f;
}
