#include <stdio.h>
#include "optimizer.h"
#include "objective.h"

#define DIMENSION 100  // Must be even for Rosenbrock

int main() {
    double x[DIMENSION];

    // Initialize starting point: typical for Rosenbrock is (-1.2, 1.0, -1.2, 1.0, ...)
    for (size_t i = 0; i < DIMENSION; i += 2) {
        x[i] = -1.2;
        x[i + 1] = 1.0;
    }

    // For full gradients, we call init_sparse_mask with sparsity = 1.0.
    init_sparse_mask(DIMENSION, 1.0);

    LBFGSParams params;
    params.max_iterations = 100;
    params.epsilon = 1e-6;
    params.history_size = 10;

    printf("Starting L-BFGS optimization on the Rosenbrock function...\n");
    size_t iterations = lbfgs_optimize(x, DIMENSION, objective_function, params);
    printf("Optimization finished in %zu iterations.\n", iterations);

    printf("Final solution (first 10 elements):\n");
    for (size_t i = 0; i < 10; i++) {
        printf("%.5f ", x[i]);
    }
    printf("\n");

    return 0;
}
