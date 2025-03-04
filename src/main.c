#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "pde_solver.h"

#define NX 300
#define NY 300
#define NSIM 100  // Number of simulations for the demonstration

int main() {
    int nx = NX;
    int ny = NY;
    int grid_size = nx * ny;
    
    double *k = (double *)malloc(grid_size * sizeof(double));
    double *Q = (double *)malloc(grid_size * sizeof(double));
    double *T = (double *)malloc(grid_size * sizeof(double));
    if (!k || !Q || !T) {
        fprintf(stderr, "Memory allocation failed in main.\n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            int idx = i * nx + j;
            if (i > ny * 0.4 && i < ny * 0.6 && j > nx * 0.4 && j < nx * 0.6)
                Q[idx] = 100.0;
            else
                Q[idx] = 0.0;
        }
    }
    
    srand((unsigned)time(NULL));
    
    clock_t start = clock();
    for (int sim = 0; sim < NSIM; sim++) {
        for (int i = 0; i < grid_size; i++) {
            k[i] = 0.1 + 0.9 * ((double)rand() / RAND_MAX);
        }
        int iterations = solve_pde(k, Q, T, nx, ny, 5000, 1e-6);
        if (sim == 0) {
            printf("Simulation %d: Converged in %d iterations.\n", sim + 1, iterations);
        }
    }
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Completed %d simulations in %.3f seconds.\n", NSIM, elapsed);
    
    double sumT = 0.0;
    for (int i = 0; i < grid_size; i++) {
        sumT += T[i];
    }
    printf("Average temperature (last simulation): %.5f\n", sumT / grid_size);
    
    free(k);
    free(Q);
    free(T);
    
    return 0;
}
