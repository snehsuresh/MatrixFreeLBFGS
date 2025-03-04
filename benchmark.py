#!/usr/bin/env python3
"""
Benchmarking SciPy's L-BFGS on the same sparse quadratic objective.
The objective: f(x) = 0.5 * sum_{i in S} (x[i]-1)^2, where S is a random subset.
"""

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import time

# Set dimension and sparsity.
DIMENSION = 1000
SPARSITY = 0.1  # fraction of indices that are active

# Generate a random mask for sparsity.
np.random.seed(42)
mask = (np.random.rand(DIMENSION) < SPARSITY).astype(float)


def objective(x):
    # Compute only for active indices.
    diff = x - 1.0
    f = 0.5 * np.sum(mask * diff * diff)
    return f


def grad(x):
    diff = x - 1.0
    g = mask * diff
    return g


# Callback to record progress.
history = {"f": []}


def callback(xk):
    f_val = objective(xk)
    history["f"].append(f_val)


def main():
    x0 = np.zeros(DIMENSION)
    start_time = time.time()
    res = opt.minimize(
        objective,
        x0,
        jac=grad,
        method="L-BFGS-B",
        callback=callback,
        options={"disp": True, "maxiter": 100},
    )
    elapsed = time.time() - start_time
    print(
        "SciPy L-BFGS finished in {:.4f} seconds, f(x) = {:.6f}".format(
            elapsed, res.fun
        )
    )

    # Plot convergence.
    plt.figure()
    plt.plot(history["f"], marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Function Value")
    plt.title("SciPy L-BFGS Convergence")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
