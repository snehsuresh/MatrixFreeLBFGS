#include <stdio.h>
#include "optimizer.h"
#include "objective.h"
#include <time.h>

#define DIMENSION 100  // Must be even

int main() {
    double x[DIMENSION];
    // Typical Rosenbrock starting point: (-1.2, 1.0, -1.2, 1.0, ...)
    for (size_t i = 0; i < DIMENSION; i += 2) {
        x[i] = -1.2;
        x[i + 1] = 1.0;
    }
    
    // For full gradient mode, call init_sparse_mask with sparsity = 1.0 (does nothing here).
    init_sparse_mask(DIMENSION, 1.0);
    
    LBFGSParams params;
    params.max_iterations = 100;
    params.tolerance = 1e-6;
    params.history_size = 10;
    params.c1 = 1e-4;
    params.tau = 0.5;
    
    printf("Starting L-BFGS optimization on the Rosenbrock function...\n");
    clock_t start = clock();
    size_t iterations = lbfgs_optimize(x, DIMENSION, objective_function, params);
    clock_t end = clock();

    double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("C L-BFGS Optimization finished in %.6f seconds.\n", elapsed_time);
    printf("Optimization finished in %zu iterations.\n", iterations);
    
    printf("Final solution (first 10 elements):\n");
    for (size_t i = 0; i < 10; i++) {
        printf("%.5f ", x[i]);
    }
    printf("\n");
    
    return 0;
}
