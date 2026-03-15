"""
Velocity and Acceleration Analysis for 2D Multibody Systems
============================================================
After solving the position problem (F(q) = 0), the same Jacobian Fq
is reused to solve the linear velocity and acceleration equations:

    Velocity:     Fq · q̇ = b_v
    Acceleration: Fq · q̈ = b_a

See Nikravesh (1988), Chapter 3.2.

Usage
-----
    from utils.kinematics import velocity_analysis, acceleration_analysis

    # After solving position:
    F, Jac = constraint_fn(q)

    # Velocity (provide time-derivatives of driven constraints RHS):
    qdot = velocity_analysis(Jac, b_v)

    # Acceleration (provide second time-derivatives of driven constraints RHS):
    qdotdot = acceleration_analysis(q, qdot, Jac, constraint_fn, b_a)
"""

import numpy as np


def velocity_analysis(Jac, b_v):
    """
    Solve the velocity equation: Fq · q̇ = b_v

    For a typical crank-driven mechanism:
        - Rows for driven-angle constraint contribute b_v[row] = θ̇_driven
        - All other constraint rows contribute 0 (scleronomous constraints)

    Parameters
    ----------
    Jac : (nc, n) Jacobian matrix  (from constraint_fn at current q)
    b_v : (nc,)   right-hand side  (usually sparse; non-zero only at driven DOF rows)

    Returns
    -------
    qdot : (n,) velocity vector  (least-squares solution)
    """
    qdot, *_ = np.linalg.lstsq(Jac, b_v, rcond=None)
    return qdot


def acceleration_analysis(q, qdot, Jac, constraint_fn, b_a, eps=1e-7):
    """
    Solve the acceleration equation: Fq · q̈ = b_a - γ

    where γ is the quadratic velocity term:
        γ = (d/dt Fq) · q̇  ≈  [Fq(q + ε·qdot·dt) - Fq(q)] / dt · qdot

    computed here by finite differences on the Jacobian.

    For a typical crank-driven mechanism:
        - Rows for driven-angle constraint contribute b_a[row] = θ̈_driven
        - All other rows contribute 0

    Parameters
    ----------
    q            : (n,)   current position vector
    qdot         : (n,)   current velocity vector
    Jac          : (nc, n) current Jacobian  (from constraint_fn at q)
    constraint_fn: callable, q → (F, Jac)   used for finite-difference of Jac
    b_a          : (nc,)   RHS (non-zero only at rows with time-varying constraints)
    eps          : float,  finite-difference step size (default 1e-7)

    Returns
    -------
    qdotdot : (n,) acceleration vector
    """
    # Approximate (d/dt Fq) · qdot  via finite differences:
    #   γ ≈ [Fq(q + eps*qdot) - Fq(q)] / eps · qdot  — but since we want Jac_dot · qdot,
    #   it is cleaner to do:   γ = [Fq(q + eps*qdot) · qdot - Fq(q) · qdot] / eps
    _, Jac_pert = constraint_fn(q + eps * qdot)
    gamma = (Jac_pert @ qdot - Jac @ qdot) / eps

    rhs = b_a - gamma
    qdotdot, *_ = np.linalg.lstsq(Jac, rhs, rcond=None)
    return qdotdot


def build_b_velocity(nc, driven_rows, theta_dot_driven):
    """
    Build the RHS vector for velocity analysis.

    Parameters
    ----------
    nc              : int, total number of constraint equations
    driven_rows     : list of int, row indices corresponding to driving constraints
    theta_dot_driven: list of float, prescribed angular velocities for each driven row

    Returns
    -------
    b_v : (nc,) RHS vector
    """
    b_v = np.zeros(nc)
    for row, val in zip(driven_rows, theta_dot_driven):
        b_v[row] = val
    return b_v


def build_b_acceleration(nc, driven_rows, theta_dotdot_driven):
    """
    Build the RHS vector for acceleration analysis.

    Parameters
    ----------
    nc                  : int, total number of constraint equations
    driven_rows         : list of int, row indices of driving constraints
    theta_dotdot_driven : list of float, prescribed angular accelerations

    Returns
    -------
    b_a : (nc,) RHS vector
    """
    b_a = np.zeros(nc)
    for row, val in zip(driven_rows, theta_dotdot_driven):
        b_a[row] = val
    return b_a


def nullspace_basis(Jac, tol=1e-10):
    """
    Compute an orthonormal basis for the nullspace of Fq (subspace of allowable motions).

    Uses SVD:  Fq = U S V^T  →  nullspace = columns of V where S ≈ 0.

    The nullspace gives the independent velocity directions the mechanism
    can have without violating constraints — i.e., its mobility (DOF).

    Parameters
    ----------
    Jac : (nc, n) Jacobian
    tol : float, singular-value threshold for zero (default 1e-10)

    Returns
    -------
    R : (n, dof) matrix whose columns span the nullspace of Jac
        dof = n - rank(Fq)
    """
    _, S, Vt = np.linalg.svd(Jac, full_matrices=True)
    rank = np.sum(S > tol)
    return Vt[rank:].T  # columns of V corresponding to zero singular values
