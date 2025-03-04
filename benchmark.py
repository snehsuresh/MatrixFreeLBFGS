#!/usr/bin/env python3
import numpy as np
import time
import math


def solve_pde_py(k, Q, nx, ny, max_iter=5000, tol=1e-6):
    # Initialize T with zeros (Dirichlet boundary conditions).
    T = np.zeros((ny, nx))
    h = 1.0 / (nx - 1)
    h2 = h * h
    for it in range(max_iter):
        T_new = T.copy()
        res_sq = 0.0
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                # Compute half-node conductivity (average)
                k_e = (k[i, j] + k[i, j + 1]) * 0.5
                k_w = (k[i, j] + k[i, j - 1]) * 0.5
                k_n = (k[i, j] + k[i - 1, j]) * 0.5
                k_s = (k[i, j] + k[i + 1, j]) * 0.5
                aij = k_e + k_w + k_n + k_s
                rhs = (
                    k_e * T[i, j + 1]
                    + k_w * T[i, j - 1]
                    + k_n * T[i - 1, j]
                    + k_s * T[i + 1, j]
                    + h2 * Q[i, j]
                )
                T_new[i, j] = rhs / aij
                diff = T_new[i, j] - T[i, j]
                res_sq += diff * diff
        T = T_new
        if math.sqrt(res_sq) < tol:
            break
    return T, it


def main():
    nx = 300
    ny = 300
    NSIM = 10  # For Python, use fewer simulations because it's very slow in pure loops.
    Q = np.zeros((ny, nx))
    # Set Q: nonzero in central region.
    for i in range(ny):
        for j in range(nx):
            if i > ny * 0.4 and i < ny * 0.6 and j > nx * 0.4 and j < nx * 0.6:
                Q[i, j] = 100.0
    start = time.time()
    for sim in range(NSIM):
        # Generate random conductivity field between 0.1 and 1.0.
        k = 0.1 + 0.9 * np.random.rand(ny, nx)
        T, iters = solve_pde_py(k, Q, nx, ny)
        if sim == 0:
            print(f"Simulation {sim+1}: Converged in {iters} iterations.")
    elapsed = time.time() - start
    print(f"Completed {NSIM} simulations in {elapsed:.3f} seconds.")
    print(f"Average temperature (last simulation): {np.mean(T):.5f}")


if __name__ == "__main__":
    main()
