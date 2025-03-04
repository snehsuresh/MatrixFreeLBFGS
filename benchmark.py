#!/usr/bin/env python3
"""
Benchmarking SciPy's L-BFGS-B on the Rosenbrock function.
"""
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import time

DIMENSION = 100  # Must be even


def rosenbrock(x):
    f = 0.0
    for i in range(0, len(x), 2):
        x1 = x[i]
        x2 = x[i + 1]
        f += (1 - x1) ** 2 + 100 * (x2 - x1**2) ** 2
    return f


def rosenbrock_grad(x):
    grad = np.zeros_like(x)
    for i in range(0, len(x), 2):
        x1 = x[i]
        x2 = x[i + 1]
        grad[i] = -2 * (1 - x1) - 400 * x1 * (x2 - x1**2)
        grad[i + 1] = 200 * (x2 - x1**2)
    return grad


history = []


def callback(xk):
    history.append(rosenbrock(xk))


def main():
    x0 = np.empty(DIMENSION)
    for i in range(0, DIMENSION, 2):
        x0[i] = -1.2
        x0[i + 1] = 1.0

    start = time.time()
    res = opt.minimize(
        rosenbrock,
        x0,
        jac=rosenbrock_grad,
        method="L-BFGS-B",
        callback=callback,
        options={"disp": True, "maxiter": 100},
    )
    elapsed = time.time() - start
    print(
        "SciPy L-BFGS-B finished in {:.4f} seconds, f = {:.6f}".format(elapsed, res.fun)
    )

    plt.figure()
    plt.plot(history, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Objective value")
    plt.title("SciPy L-BFGS-B Convergence on Rosenbrock")
    plt.grid(True)
    # plt.show()
    plt.savefig("python_lbfgs_convergence.png", dpi=300)


if __name__ == "__main__":
    main()
