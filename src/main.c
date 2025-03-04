#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "optimizer.h"
#include "objective.h"

#define DIMENSION 1000

int main() {
    // Seed random number generator.
    srand((unsigned int)time(NULL));

    // Allocate initial guess x.
    double *x = (double *)malloc(DIMENSION * sizeof(double));
    // Initialize x to zeros.
    for (size_t i = 0; i < DIMENSION; i++) {
        x[i] = 0.0;
    }

    // Initialize sparse mask with 10% active entries.
    init_sparse_mask(DIMENSION, 0.1);

    // Set L-BFGS parameters.
    LBFGSParams params;
    params.m = 10;
    params.max_iter = 100;
    params.tol = 1e-6;
    params.alpha_init = 1.0;
    params.c1 = 1e-4;
    params.tau = 0.5;
    params.checkpoint_freq = 10;

    printf("Starting L-BFGS optimization on a sparse quadratic objective...\n");

    size_t iters = lbfgs_optimize(x, DIMENSION, objective_function, params);
    printf("Optimization finished in %zu iterations.\n", iters);
    printf("Final solution (first 10 elements):\n");
    for (size_t i = 0; i < 10; i++) {
        printf("%.6f ", x[i]);
    }
    printf("\n");

    free(x);
    return 0;
}
