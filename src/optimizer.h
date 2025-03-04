#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <stddef.h>

// Function pointer type for objective function and gradient evaluation.
// Given input x of length n, compute f(x) and gradient in grad (array of length n).
typedef double (*ObjectiveFunc)(const double *x, double *grad, size_t n);

// L-BFGS optimizer parameters structure.
typedef struct {
    size_t m;             // Maximum number of corrections (history size)
    size_t max_iter;      // Maximum number of iterations
    double tol;           // Tolerance for convergence (gradient norm)
    double alpha_init;    // Initial step size for line search
    double c1;            // Armijo condition constant
    double tau;           // Backtracking line search reduction factor
    size_t checkpoint_freq; // Frequency (in iterations) for checkpointing
} LBFGSParams;

// Main optimization function.
// Inputs:
//   x: initial guess (also used to return optimized solution)
//   n: dimension of the problem
//   obj: objective function pointer
//   params: L-BFGS parameters
// Returns the number of iterations performed.
size_t lbfgs_optimize(double *x, size_t n, ObjectiveFunc obj, LBFGSParams params);

#endif // OPTIMIZER_H
