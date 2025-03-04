Project: Matrix-Free L-BFGS Optimization in C

Overview:
This project implements a matrix-free L-BFGS optimization algorithm in C, specifically optimized for the 1000-dimensional Rosenbrock function.

Key Features:
- Matrix-free L-BFGS implementation.
- Objective function: High-dimensional Rosenbrock.
- Fully parallelized gradient computation using OpenMP.
- Supports flexible convergence criteria.
- Lightweight design: No external libraries required (other than OpenMP).

Performance Benchmark:
- Compared against SciPy's L-BFGS-B implementation.
- Test case: 1000-dimensional Rosenbrock function.
- Results:
    - C Implementation: 0.000316 seconds
    - SciPy Implementation: 0.0101 seconds
- Speedup: ~32x faster than Python baseline.

Convergence Plots:
### C Implementation Convergence
![C L-BFGS Convergence](c_lbfgs_convergence.png)

### SciPy Implementation Convergence
![SciPy L-BFGS Convergence](python_lbfgs_convergence.png)

These plots demonstrate the objective function reduction over iterations for both implementations.

Directory Structure:
- src/
    - main.c          # Entry point and driver
    - optimizer.c     # L-BFGS implementation
    - objective.c     # Rosenbrock objective and gradient computation
    - optimizer.h     # Header for L-BFGS API
    - objective.h     # Header for objective function API
- Makefile            # Build automation
- plot_lbfgs_c.py      # Plot convergence from C output
- benchmark.py         # Run SciPy L-BFGS-B for comparison

How to Build and Run:
1. Clean previous builds:
    make clean

2. Compile and link:
    make

3. Run the optimizer:
    ./lbfgs_optimizer

4. Example output:
    Starting L-BFGS optimization on the Rosenbrock function...
    [DEBUG] Initial f = 1210.0000000000, ||grad|| = 1.6466232113e+03
    ...
    Optimization finished in 39 iterations.
    Final solution (first 10 elements): 1.00000 1.00000 ...

5. Time measurement:
    Execution time is printed directly to the terminal.

Comparisons with SciPy (Python):
- Benchmark script: benchmark.py (separate Python file - not part of this repo)
- Compare runtime with SciPy’s L-BFGS-B on the same problem.
- Example result:
    SciPy L-BFGS-B finished in 0.0101 seconds, f = 0.000000
    C L-BFGS Optimization finished in 0.000316 seconds.

System Requirements:
- macOS (tested on Apple Silicon - M2)
- clang with OpenMP support
- Make utility
- Python (for plotting)

Technical Highlights:
- Matrix-free formulation avoids explicit inverse Hessian storage.
- OpenMP parallelism for gradient computation.
- Minimal overhead through low-level memory management.
- Direct replacement for Python-based L-BFGS where performance matters.

