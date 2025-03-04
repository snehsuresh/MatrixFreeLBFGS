#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "pde_solver.h"

#define INDEX(i,j,nx) ((i)*(nx) + (j))

int solve_pde(const double *k, const double *Q, double *T, int nx, int ny, int max_iter, double tol) {
    int iter;
    double h2 = 1.0 / ((nx - 1) * (ny - 1));
    double *T_new = (double *)malloc(nx * ny * sizeof(double));
    if (!T_new) {
        fprintf(stderr, "Memory allocation failed in solver.\n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < nx * ny; i++) T[i] = 0.0;
    
    for (iter = 0; iter < max_iter; iter++) {
        double res_sq = 0.0;
        
        for (int i = 1; i < ny - 1; i++) {
            for (int j = 1; j < nx - 1; j++) {
                int idx = INDEX(i,j,nx);
                double k_e = 0.5 * (k[idx] + k[INDEX(i, j+1, nx)]);
                double k_w = 0.5 * (k[idx] + k[INDEX(i, j-1, nx)]);
                double k_n = 0.5 * (k[idx] + k[INDEX(i-1, j, nx)]);
                double k_s = 0.5 * (k[idx] + k[INDEX(i+1, j, nx)]);
                double aij = k_e + k_w + k_n + k_s;
                
                double rhs = k_e * T[INDEX(i, j+1, nx)] +
                             k_w * T[INDEX(i, j-1, nx)] +
                             k_n * T[INDEX(i-1, j, nx)] +
                             k_s * T[INDEX(i+1, j, nx)] +
                             h2 * Q[idx];
                             
                T_new[idx] = rhs / aij;
                double diff = T_new[idx] - T[idx];
                res_sq += diff * diff;
            }
        }
        
        for (int i = 1; i < ny - 1; i++)
            for (int j = 1; j < nx - 1; j++)
                T[INDEX(i,j,nx)] = T_new[INDEX(i,j,nx)];
        
        if (sqrt(res_sq) < tol) break;
    }
    
    free(T_new);
    return iter;
}
