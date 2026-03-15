"""
Newton-Raphson Solver for Multibody Position Problems
=====================================================
Solves the nonlinear constraint system F(q) = 0 using Newton-Raphson
iteration with a least-squares update step, which handles redundant
constraints gracefully (see Ch. 3.4 of Nikravesh 1988).

Usage
-----
    from utils.solver import newton_raphson
    from utils.constraints_2d import assemble, revolute_constraint, ...

    def my_constraints(q):
        return assemble([
            revolute_constraint(q, ...),
            ...
        ])

    q_sol, converged = newton_raphson(q0, my_constraints)
"""

import numpy as np


def newton_raphson(q0, constraint_fn, tol=1e-10, max_iter=50, verbose=False):
    """
    Solve F(q) = 0 using Newton-Raphson iteration.

    Update rule (least-squares, handles redundant constraints):
        Fq(qᵢ) Δq = -F(qᵢ)
        qᵢ₊₁ = qᵢ + Δq

    The least-squares solve (np.linalg.lstsq) gives the minimum-norm
    correction when the system is over-determined or rank-deficient,
    consistent with Eq. 3.15 of Nikravesh (1988).

    Parameters
    ----------
    q0            : (n,) array-like, initial guess for the coordinate vector
    constraint_fn : callable, q → (F, Jac)
                    Must return:
                        F   : (nc,)   residual vector
                        Jac : (nc, n) Jacobian matrix
    tol           : float, convergence tolerance on ||F||_inf (default 1e-10)
    max_iter      : int,   maximum number of iterations  (default 50)
    verbose       : bool,  print iteration info          (default False)

    Returns
    -------
    q         : (n,) solution coordinate vector
    converged : bool, True if ||F||_inf < tol at termination
    """
    q = np.array(q0, dtype=float)

    for k in range(max_iter):
        F, Jac = constraint_fn(q)
        err = np.max(np.abs(F))

        if verbose:
            print(f"  iter {k:3d}  ||F||_inf = {err:.3e}")

        if err < tol:
            return q, True

        # Δq = -Fq \ F  (least-squares for robustness)
        dq, *_ = np.linalg.lstsq(Jac, -F, rcond=None)
        q = q + dq

    # Final check
    F, _ = constraint_fn(q)
    err = np.max(np.abs(F))
    if verbose:
        print(f"  iter {max_iter:3d}  ||F||_inf = {err:.3e}  (max iterations reached)")

    return q, err < tol


def newton_raphson_trajectory(q0, constraint_fn_factory, param_values,
                              tol=1e-10, max_iter=50, verbose=False):
    """
    Solve a sequence of position problems along a parameter trajectory
    (e.g., crank angle sweep), using each solution as the initial guess
    for the next step.

    Parameters
    ----------
    q0                  : (n,) initial coordinate guess for first step
    constraint_fn_factory : callable, param → constraint_fn
                            Returns a constraint function for the given parameter
                            value (e.g., driving angle).
    param_values        : iterable of parameter values to sweep over
    tol, max_iter, verbose : passed to newton_raphson

    Returns
    -------
    q_traj   : (n_steps, n) array of solutions, one row per parameter value
    converged: (n_steps,)  bool array
    """
    param_values = list(param_values)
    n = len(q0)
    n_steps = len(param_values)

    q_traj    = np.zeros((n_steps, n))
    converged = np.zeros(n_steps, dtype=bool)

    q = np.array(q0, dtype=float)
    for k, param in enumerate(param_values):
        fn = constraint_fn_factory(param)
        q, ok = newton_raphson(q, fn, tol=tol, max_iter=max_iter, verbose=verbose)
        q_traj[k]    = q
        converged[k] = ok
        if not ok and verbose:
            print(f"  WARNING: did not converge at param={param}")

    return q_traj, converged
