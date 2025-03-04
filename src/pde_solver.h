#ifndef PDE_SOLVER_H
#define PDE_SOLVER_H

// Solve the variable-coefficient Poisson equation (steady-state heat conduction)
// on a 2D grid with Dirichlet boundary conditions (T=0 on the boundary).
// The PDE is discretized as:
//   -div(k * grad T) = Q,
// where the half-node conductivity is approximated by averaging.
// 
// Parameters:
//   k      : conductivity field (array of size nx*ny)
//   Q      : source term (array of size nx*ny)
//   T      : output temperature field (array of size nx*ny)
//   nx,ny  : grid dimensions
//   max_iter: maximum Jacobi iterations
//   tol    : convergence tolerance (L2 norm of update)
// Returns:
//   number of iterations performed.
int solve_pde(const double *k, const double *Q, double *T, int nx, int ny, int max_iter, double tol);

#endif // PDE_SOLVER_H
