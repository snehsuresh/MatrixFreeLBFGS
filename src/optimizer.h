#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <stddef.h>

// Function pointer type for objective evaluation.
// Given input x (array of length n), compute objective and gradient (array of length n).
typedef double (*ObjectiveFunc)(const double *x, double *grad, size_t n);

// LBFGS parameters structure.
typedef struct {
    size_t max_iterations;  // Maximum iterations
    double epsilon;         // Convergence tolerance on gradient norm
    size_t history_size;    // Memory for L-BFGS (number of stored updates)
} LBFGSParams;

// Main L-BFGS optimizer function.
// x: input initial guess (also returns the optimized solution)
// n: problem dimension
// obj: pointer to objective function (which computes both f and grad)
// params: LBFGS parameters
// Returns the number of iterations performed.
size_t lbfgs_optimize(double *x, size_t n, ObjectiveFunc obj, LBFGSParams params);

#endif // OPTIMIZER_H
