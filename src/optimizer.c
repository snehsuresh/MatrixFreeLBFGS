#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "optimizer.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// Save checkpoint to file.
static void save_checkpoint(const double *x, size_t n, size_t iter) {
    FILE *fp = fopen("checkpoint.txt", "w");
    if (!fp) {
        fprintf(stderr, "Error writing checkpoint file.\n");
        return;
    }
    fprintf(fp, "# Iteration %zu\n", iter);
    for (size_t i = 0; i < n; i++) {
        fprintf(fp, "%.10f ", x[i]);
    }
    fprintf(fp, "\n");
    fclose(fp);
}

// Load checkpoint from file if exists. Returns 1 if loaded, 0 otherwise.
static int load_checkpoint(double *x, size_t n) {
    FILE *fp = fopen("checkpoint.txt", "r");
    if (!fp) return 0;
    char line[256];
    if (!fgets(line, sizeof(line), fp)) {
        fclose(fp);
        return 0;
    }
    // Read vector values.
    for (size_t i = 0; i < n; i++) {
        if (fscanf(fp, "%lf", &x[i]) != 1) {
            fclose(fp);
            return 0;
        }
    }
    fclose(fp);
    return 1;
}

// Dot product of two vectors.
static double dot(const double *a, const double *b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Compute L2 norm of a vector.
static double norm(const double *a, size_t n) {
    return sqrt(dot(a, a, n));
}

// Copy vector src to dst.
static void copy_vector(const double *src, double *dst, size_t n) {
    memcpy(dst, src, n * sizeof(double));
}

// Two-loop recursion for L-BFGS to compute search direction.
// Uses history arrays: s_list and y_list, each of size history_count x n.
// q is input gradient, returns direction in 'direction'.
static void two_loop_recursion(const double *q, double *direction, 
                               double **s_list, double **y_list, size_t history_count, size_t n) {
    double *alpha = (double *)malloc(history_count * sizeof(double));
    double *rho = (double *)malloc(history_count * sizeof(double));
    double *q_vec = (double *)malloc(n * sizeof(double));
    copy_vector(q, q_vec, n);

    for (size_t i = history_count; i-- > 0;) {
        rho[i] = 1.0 / dot(y_list[i], s_list[i], n);
        alpha[i] = rho[i] * dot(s_list[i], q_vec, n);
        for (size_t j = 0; j < n; j++) {
            q_vec[j] -= alpha[i] * y_list[i][j];
        }
    }
    // Use scaling of initial Hessian: gamma = (s_{k-1}^T y_{k-1}) / (y_{k-1}^T y_{k-1})
    double gamma = 1.0;
    if (history_count > 0) {
        double sy = dot(s_list[history_count - 1], y_list[history_count - 1], n);
        double yy = dot(y_list[history_count - 1], y_list[history_count - 1], n);
        gamma = sy / yy;
    }
    for (size_t i = 0; i < n; i++) {
        direction[i] = gamma * q_vec[i];
    }
    for (size_t i = 0; i < history_count; i++) {
        double beta = rho[i] * dot(y_list[i], direction, n);
        for (size_t j = 0; j < n; j++) {
            direction[j] += s_list[i][j] * (alpha[i] - beta);
        }
    }
    free(alpha);
    free(rho);
    free(q_vec);
}

// Backtracking line search with Armijo condition.
static double line_search(double *x, double *f, double *grad, const double *direction, size_t n, 
                            ObjectiveFunc obj, double alpha_init, double c1, double tau) {
    double alpha = alpha_init;
    double f_old = *f;
    double *x_new = (double *)malloc(n * sizeof(double));
    double *grad_new = (double *)malloc(n * sizeof(double));
    while (1) {
        for (size_t i = 0; i < n; i++) {
            x_new[i] = x[i] + alpha * direction[i];
        }
        double f_new = obj(x_new, grad_new, n);
        // Armijo condition: f(x+alpha*p) <= f(x) + c1 * alpha * grad^T * p
        double dot_gp = dot(grad, direction, n);
        if (f_new <= f_old + c1 * alpha * dot_gp) {
            *f = f_new;
            copy_vector(grad_new, grad, n);
            break;
        }
        alpha *= tau;
        if (alpha < 1e-10) {
            // Step size too small.
            break;
        }
    }
    free(x_new);
    free(grad_new);
    return alpha;
}

// Main L-BFGS optimizer function.
size_t lbfgs_optimize(double *x, size_t n, ObjectiveFunc obj, LBFGSParams params) {
    size_t iter = 0;
    double f_val;
    double *grad = (double *)malloc(n * sizeof(double));
    f_val = obj(x, grad, n);

    // Allocate history arrays for s and y vectors.
    double **s_list = (double **)malloc(params.m * sizeof(double *));
    double **y_list = (double **)malloc(params.m * sizeof(double *));
    for (size_t i = 0; i < params.m; i++) {
        s_list[i] = (double *)calloc(n, sizeof(double));
        y_list[i] = (double *)calloc(n, sizeof(double));
    }
    size_t history_count = 0;

    double *direction = (double *)malloc(n * sizeof(double));
    double *x_old = (double *)malloc(n * sizeof(double));
    double *grad_old = (double *)malloc(n * sizeof(double));

    // Check if there is a checkpoint.
    if (load_checkpoint(x, n)) {
        printf("Checkpoint loaded. Resuming from saved state.\n");
        f_val = obj(x, grad, n);
    }

    while (iter < params.max_iter && norm(grad, n) > params.tol) {
        // Compute search direction using two-loop recursion.
        if (history_count == 0) {
            // First iteration: use negative gradient.
            for (size_t i = 0; i < n; i++) {
                direction[i] = -grad[i];
            }
        } else {
            two_loop_recursion(grad, direction, s_list, y_list, history_count, n);
            for (size_t i = 0; i < n; i++) {
                direction[i] = -direction[i];
            }
        }

        // Save current x and gradient.
        copy_vector(x, x_old, n);
        copy_vector(grad, grad_old, n);

        // Line search to determine step size.
        double step = line_search(x, &f_val, grad, direction, n, obj, params.alpha_init, params.c1, params.tau);

        // Update s and y history.
        double *s = s_list[history_count % params.m];
        double *y = y_list[history_count % params.m];
        for (size_t i = 0; i < n; i++) {
            s[i] = x[i] - x_old[i];
            y[i] = grad[i] - grad_old[i];
        }
        if (history_count < params.m) {
            history_count++;
        } else {
            // Overwrite oldest history; already done using modulo indexing.
        }

        iter++;

        // Print progress.
        printf("Iteration %zu: f = %.10f, ||grad|| = %.10e, step = %.3e\n", iter, f_val, norm(grad, n), step);

        // Checkpointing.
        if (iter % params.checkpoint_freq == 0) {
            save_checkpoint(x, n, iter);
        }
    }

    // Clean up
    for (size_t i = 0; i < params.m; i++) {
        free(s_list[i]);
        free(y_list[i]);
    }
    free(s_list);
    free(y_list);
    free(grad);
    free(direction);
    free(x_old);
    free(grad_old);
    return iter;
}
