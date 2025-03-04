#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "optimizer.h"
#include "objective.h"

typedef struct {
    double *s;  // Difference in x (x_{k+1} - x_k)
    double *y;  // Difference in gradient (grad_{k+1} - grad_k)
    double rho; // 1.0 / (y^T s)
} LBFGS_Update;

// Helper: Compute dot product of vectors a and b of length n.
static double dot_product(const double *a, const double *b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Helper: Copy vector from src to dst.
static void copy_vector(const double *src, double *dst, size_t n) {
    memcpy(dst, src, n * sizeof(double));
}

size_t lbfgs_optimize(double *x, size_t n, ObjectiveFunc obj, LBFGSParams params) {
    size_t max_iter = params.max_iterations;
    double tol = params.tolerance;
    size_t m = params.history_size;
    double c1 = params.c1;
    double tau = params.tau;

    double *grad = (double *)malloc(n * sizeof(double));
    double *grad_prev = (double *)malloc(n * sizeof(double));
    double *x_prev = (double *)malloc(n * sizeof(double));

    double f = obj(x, grad, n);
    double grad_norm = sqrt(dot_product(grad, grad, n));
    printf("[DEBUG] Initial f = %.10f, ||grad|| = %.10e\n", f, grad_norm);

    // Allocate history storage for LBFGS updates.
    LBFGS_Update *updates = (LBFGS_Update *)malloc(m * sizeof(LBFGS_Update));
    for (size_t i = 0; i < m; i++) {
        updates[i].s = (double *)malloc(n * sizeof(double));
        updates[i].y = (double *)malloc(n * sizeof(double));
        updates[i].rho = 0.0;
    }
    size_t history_count = 0; // total updates stored
    size_t start = 0;         // index of the oldest update in the cyclic buffer

    size_t iter;
    for (iter = 0; iter < max_iter && grad_norm > tol; iter++) {
        // Save current state.
        copy_vector(x, x_prev, n);
        copy_vector(grad, grad_prev, n);

        // --- Two-Loop Recursion ---
        double *q = (double *)malloc(n * sizeof(double));
        copy_vector(grad, q, n);
        int count = (history_count < m) ? history_count : m;
        double *alpha = (double *)malloc(count * sizeof(double));
        // Loop backward over history (most recent first).
        for (int i = count - 1; i >= 0; i--) {
            int index = (start + i) % m;
            double dot = dot_product(updates[index].s, q, n);
            alpha[i] = updates[index].rho * dot;
            for (size_t j = 0; j < n; j++) {
                q[j] -= alpha[i] * updates[index].y[j];
            }
        }
        // Scaling of initial Hessian (H0).
        double H0 = 1.0;
        if (count > 0) {
            int index = (start + count - 1) % m;
            double sy = dot_product(updates[index].s, updates[index].y, n);
            double yy = dot_product(updates[index].y, updates[index].y, n);
            if (yy > 0.0) H0 = sy / yy;
        }
        for (size_t i = 0; i < n; i++) {
            q[i] *= H0;
        }
        // Loop forward.
        for (int i = 0; i < count; i++) {
            int index = (start + i) % m;
            double dot = dot_product(updates[index].y, q, n);
            double beta = updates[index].rho * dot;
            for (size_t j = 0; j < n; j++) {
                q[j] += updates[index].s[j] * (alpha[i] - beta);
            }
        }
        free(alpha);
        // Search direction is the negative of q.
        for (size_t i = 0; i < n; i++) {
            q[i] = -q[i];
        }
        // q now holds the search direction d.
        
        // --- Line Search using Armijo condition ---
        double step = 1.0;
        double *x_new = (double *)malloc(n * sizeof(double));
        double *grad_new = (double *)malloc(n * sizeof(double));
        double f_new;
        while (1) {
            // x_new = x + step * d
            for (size_t i = 0; i < n; i++) {
                x_new[i] = x[i] + step * q[i];
            }
            f_new = obj(x_new, grad_new, n);
            double gd = dot_product(grad, q, n);
            if (f_new <= f + c1 * step * gd) {
                break;
            }
            step *= tau;
            if (step < 1e-12) {
                printf("[WARN] Line search failed at iteration %zu\n", iter);
                break;
            }
        }
        free(q);

        // Update x, f, and grad.
        copy_vector(x_new, x, n);
        f = f_new;
        copy_vector(grad_new, grad, n);
        free(x_new);
        free(grad_new);

        // --- Update LBFGS History (skip first iteration) ---
        if (iter > 0) {
            double *s = (double *)malloc(n * sizeof(double));
            double *y = (double *)malloc(n * sizeof(double));
            for (size_t i = 0; i < n; i++) {
                s[i] = x[i] - x_prev[i];
                y[i] = grad[i] - grad_prev[i];
            }
            double sy = dot_product(s, y, n);
            if (sy == 0.0) sy = 1e-10;
            double rho_val = 1.0 / sy;
            int pos = history_count % m;
            copy_vector(s, updates[pos].s, n);
            copy_vector(y, updates[pos].y, n);
            updates[pos].rho = rho_val;
            if (history_count >= m) {
                start = (history_count + 1) % m;
            }
            history_count++;
            free(s);
            free(y);
        }
        
        grad_norm = sqrt(dot_product(grad, grad, n));
        printf("Iteration %zu: f = %.10f, ||grad|| = %.10e, step = %.3e\n", iter+1, f, grad_norm, step);
    }
    
    // Free LBFGS update storage.
    for (size_t i = 0; i < m; i++) {
        free(updates[i].s);
        free(updates[i].y);
    }
    free(updates);
    free(grad);
    free(grad_prev);
    free(x_prev);
    
    return iter;
}
