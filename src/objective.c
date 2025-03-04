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

    size_t active_count = 0;

    for (size_t i = 0; i < n; i++) {
        double r = (double)rand() / RAND_MAX;
        // mask[i] = (r < sparsity) ? 1 : 0;
        mask[i] = 1;  // Force full gradients
        if (mask[i]) active_count++;
    }

    printf("[DEBUG] Sparse mask initialized: %zu active out of %zu (%.2f%%)\n",
           active_count, n, 100.0 * active_count / n);
}


// Objective function: f(x) = 0.5 * sum_{i in S} (x[i] - 1)^2
// Also compute gradient: grad[i] = (x[i] - 1) if i in S, else 0.
double objective_function(const double *x, double *grad, size_t n) {
    double f = 0.0;
    // If mask is not initialized, assume full gradient.
    if (!mask) {
        fprintf(stderr, "[ERROR] Sparse mask not initialized before objective_function() call!\n");
        exit(EXIT_FAILURE);  
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
