#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <stddef.h>

// Function pointer type for objective evaluation.
// Given input x (array of length n), compute the objective and its gradient (stored in grad).
typedef double (*ObjectiveFunc)(const double *x, double *grad, size_t n);

// LBFGS parameters structure.
typedef struct {
    size_t max_iterations;  // Maximum iterations
    double tolerance;       // Convergence tolerance (norm of gradient)
    size_t history_size;    // Number of stored correction pairs
    double c1;              // Armijo condition parameter (e.g., 1e-4)
    double tau;             // Step size reduction factor (e.g., 0.5)
} LBFGSParams;

// LBFGS optimizer function.
// x: input initial guess (and returns optimized solution)
// n: dimension of the problem
// obj: pointer to objective function (computes function and gradient)
// params: LBFGS parameters
// Returns the number of iterations performed.
size_t lbfgs_optimize(double *x, size_t n, ObjectiveFunc obj, LBFGSParams params);

#endif // OPTIMIZER_H
