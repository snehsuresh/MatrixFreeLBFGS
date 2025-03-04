#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "optimizer.h"
#include "objective.h"

// Helper: Compute Euclidean norm of vector v of length n.
double vector_norm(const double *v, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += v[i] * v[i];
    }
    return sqrt(sum);
}

// Helper: Copy vector src to dest.
void vector_copy(const double *src, double *dest, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dest[i] = src[i];
    }
}

// Helper: Compute result = x + alpha * d.
void vector_add_scaled(double *result, const double *x, double alpha, const double *d, size_t n) {
    for (size_t i = 0; i < n; i++) {
        result[i] = x[i] + alpha * d[i];
    }
}

size_t lbfgs_optimize(double *x, size_t n, ObjectiveFunc obj, LBFGSParams params) {
    size_t max_iter = params.max_iterations;
    double epsilon = params.epsilon;
    size_t m = params.history_size;

    // Allocate arrays for current gradient and for previous x and gradient (for history)
    double *grad = (double *)malloc(n * sizeof(double));
    double *prev_x = (double *)malloc(n * sizeof(double));
    double *prev_grad = (double *)malloc(n * sizeof(double));

    // Allocate history arrays (s and y updates) and scaling factors rho.
    double **s_history = (double **)malloc(m * sizeof(double *));
    double **y_history = (double **)malloc(m * sizeof(double *));
    double *rho = (double *)malloc(m * sizeof(double));
    for (size_t i = 0; i < m; i++) {
        s_history[i] = (double *)malloc(n * sizeof(double));
        y_history[i] = (double *)malloc(n * sizeof(double));
    }

    // Evaluate the objective and gradient at the initial guess.
    double f = obj(x, grad, n);
    double grad_norm = vector_norm(grad, n);
    printf("[DEBUG] Initial f = %.10f, ||grad|| = %.10e\n", f, grad_norm);

    int k = 0;
    int history_count = 0;

    while (k < max_iter && grad_norm > epsilon) {
        // --- Two-loop recursion to compute search direction ---
        // Allocate temporary vector q and copy grad into it.
        double *q = (double *)malloc(n * sizeof(double));
        for (size_t i = 0; i < n; i++) {
            q[i] = grad[i];
        }
        double *alpha = (double *)malloc(m * sizeof(double));

        // Loop backward over stored history.
        for (int i = history_count - 1; i >= 0; i--) {
            double dot = 0.0;
            for (size_t j = 0; j < n; j++) {
                dot += s_history[i][j] * q[j];
            }
            alpha[i] = rho[i] * dot;
            for (size_t j = 0; j < n; j++) {
                q[j] -= alpha[i] * y_history[i][j];
            }
        }

        // Scaling: Use H0 = (sᵀy)/(yᵀy) from last update, if available.
        double scaling = 1.0;
        if (history_count > 0) {
            double sy = 0.0, yy = 0.0;
            for (size_t j = 0; j < n; j++) {
                sy += s_history[history_count-1][j] * y_history[history_count-1][j];
                yy += y_history[history_count-1][j] * y_history[history_count-1][j];
            }
            if (yy > 0.0)
                scaling = sy / yy;
        }
        for (size_t i = 0; i < n; i++) {
            q[i] *= scaling;
        }

        // Loop forward over history.
        for (int i = 0; i < history_count; i++) {
            double dot = 0.0;
            for (size_t j = 0; j < n; j++) {
                dot += y_history[i][j] * q[j];
            }
            double beta = rho[i] * dot;
            for (size_t j = 0; j < n; j++) {
                q[j] += s_history[i][j] * (alpha[i] - beta);
            }
        }
        free(alpha);

        // Set search direction d = -q.
        double *direction = (double *)malloc(n * sizeof(double));
        for (size_t i = 0; i < n; i++) {
            direction[i] = -q[i];
        }
        free(q);

        // Save current x and grad for history update.
        vector_copy(x, prev_x, n);
        vector_copy(grad, prev_grad, n);

        // --- Line Search: Basic backtracking ---
        double step = 1.0;
        double *x_new = (double *)malloc(n * sizeof(double));
        double *grad_new = (double *)malloc(n * sizeof(double));
        double f_new;
        while (1) {
            vector_add_scaled(x_new, x, step, direction, n);
            f_new = obj(x_new, grad_new, n);
            if (f_new < f) {
                break;
            }
            step *= 0.5;
            if (step < 1e-12) {
                printf("[WARN] Line search failed at iteration %d\n", k);
                break;
            }
        }
        // Update solution and objective.
        vector_copy(x_new, x, n);
        f = f_new;
        vector_copy(grad_new, grad, n);
        free(x_new);
        free(grad_new);
        free(direction);

        // --- Update History ---
        if (k > 0) {
            // Compute s = x - prev_x and y = grad - prev_grad.
            double *s = s_history[history_count % m];
            double *y = y_history[history_count % m];
            for (size_t i = 0; i < n; i++) {
                s[i] = x[i] - prev_x[i];
                y[i] = grad[i] - prev_grad[i];
            }
            double dot_sy = 0.0;
            for (size_t i = 0; i < n; i++) {
                dot_sy += s[i] * y[i];
            }
            if (dot_sy == 0.0) dot_sy = 1e-10;
            rho[history_count % m] = 1.0 / dot_sy;
            if (history_count < m) history_count++;
        }

        grad_norm = vector_norm(grad, n);
        printf("Iteration %d: f = %.10f, ||grad|| = %.10e, step = %.3e\n", k+1, f, grad_norm, step);
        k++;
    }

    // Free allocated memory.
    free(grad);
    free(prev_x);
    free(prev_grad);
    for (size_t i = 0; i < m; i++) {
        free(s_history[i]);
        free(y_history[i]);
    }
    free(s_history);
    free(y_history);
    free(rho);

    return k;
}
