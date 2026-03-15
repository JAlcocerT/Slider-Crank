"""
2D Multibody Constraint Functions with Analytical Jacobians
============================================================
Coordinate layout: q = [Rx_1, Ry_1, θ_1, Rx_2, Ry_2, θ_2, ..., Rx_n, Ry_n, θ_n]
Body i (1-based) occupies indices [3*(i-1), 3*(i-1)+1, 3*(i-1)+2].

Each constraint function returns:
    F   : numpy array of constraint residuals, shape (nc,)
    Jac : numpy array of Jacobian rows,         shape (nc, 3*n_bodies)

Use `assemble()` to stack multiple constraints before passing to the solver.

Reference:
    Nikravesh, Computer-Aided Analysis of Mechanical Systems (1988)
    Chapter 3 — Kinematic Analysis
"""

import numpy as np


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _A(theta):
    """2×2 rotation matrix A(θ)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])


def _dA(theta):
    """Derivative of rotation matrix dA/dθ."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[-s, -c],
                     [ c, -s]])


def _idx(body, offset=0):
    """Return the column index of Rx, Ry, or θ for body i (1-based)."""
    return 3 * (body - 1) + offset


# ---------------------------------------------------------------------------
# Ground constraint  (3 equations)
# ---------------------------------------------------------------------------

def ground_constraint(q, body_i, n_bodies, Rx0=0.0, Ry0=0.0, theta0=0.0):
    """
    Fix body_i to ground position and orientation.

    Equations:
        Rx_i - Rx0  = 0
        Ry_i - Ry0  = 0
        θ_i  - θ0   = 0

    Parameters
    ----------
    q       : (3*n_bodies,) coordinate vector
    body_i  : int, 1-based body index
    n_bodies: int, total number of bodies
    Rx0, Ry0, theta0 : float, fixed position and angle (default: origin, 0 rad)

    Returns
    -------
    F   : (3,)               residual vector
    Jac : (3, 3*n_bodies)   Jacobian rows
    """
    n = 3 * n_bodies
    ix = _idx(body_i)

    F = np.array([
        q[ix]   - Rx0,
        q[ix+1] - Ry0,
        q[ix+2] - theta0
    ])

    Jac = np.zeros((3, n))
    Jac[0, ix]   = 1.0
    Jac[1, ix+1] = 1.0
    Jac[2, ix+2] = 1.0

    return F, Jac


# ---------------------------------------------------------------------------
# Revolute (pin) joint constraint  (2 equations)
# ---------------------------------------------------------------------------

def revolute_constraint(q, body_i, body_j, n_bodies, s_i, s_j):
    """
    Revolute (pin) joint between body_i and body_j.

    The joint point expressed in global frame:
        P = R_i + A(θ_i) s_i  (from body i)
        P = R_j + A(θ_j) s_j  (from body j)

    Equations (2):
        R_i + A(θ_i) s_i - R_j - A(θ_j) s_j = 0

    Parameters
    ----------
    q       : (3*n_bodies,) coordinate vector
    body_i  : int, 1-based index of first  body
    body_j  : int, 1-based index of second body
    n_bodies: int, total number of bodies
    s_i     : (2,) local coordinates of joint point on body_i
    s_j     : (2,) local coordinates of joint point on body_j

    Returns
    -------
    F   : (2,)               residual vector
    Jac : (2, 3*n_bodies)   Jacobian rows
    """
    n = 3 * n_bodies
    s_i = np.asarray(s_i, dtype=float)
    s_j = np.asarray(s_j, dtype=float)

    ii = _idx(body_i)
    ij = _idx(body_j)
    theta_i = q[ii+2]
    theta_j = q[ij+2]

    R_i = q[ii:ii+2]
    R_j = q[ij:ij+2]

    F = R_i + _A(theta_i) @ s_i - R_j - _A(theta_j) @ s_j

    Jac = np.zeros((2, n))
    # ∂/∂R_i  →  +I
    Jac[0, ii]   =  1.0
    Jac[1, ii+1] =  1.0
    # ∂/∂θ_i  →  +dA(θ_i) s_i
    dA_si = _dA(theta_i) @ s_i
    Jac[0, ii+2] =  dA_si[0]
    Jac[1, ii+2] =  dA_si[1]
    # ∂/∂R_j  →  -I
    Jac[0, ij]   = -1.0
    Jac[1, ij+1] = -1.0
    # ∂/∂θ_j  →  -dA(θ_j) s_j
    dA_sj = _dA(theta_j) @ s_j
    Jac[0, ij+2] = -dA_sj[0]
    Jac[1, ij+2] = -dA_sj[1]

    return F, Jac


# ---------------------------------------------------------------------------
# Driving angle constraint  (1 equation)
# ---------------------------------------------------------------------------

def driving_angle_constraint(q, body_i, n_bodies, theta_driven):
    """
    Prescribe the absolute orientation of body_i.

    Equation (1):
        θ_i - theta_driven = 0

    Parameters
    ----------
    q            : (3*n_bodies,) coordinate vector
    body_i       : int, 1-based body index
    n_bodies     : int, total number of bodies
    theta_driven : float, prescribed angle in radians

    Returns
    -------
    F   : (1,)               residual vector
    Jac : (1, 3*n_bodies)   Jacobian rows
    """
    n = 3 * n_bodies
    ix = _idx(body_i)

    F = np.array([q[ix+2] - theta_driven])

    Jac = np.zeros((1, n))
    Jac[0, ix+2] = 1.0

    return F, Jac


# ---------------------------------------------------------------------------
# Relative angle (prismatic orientation) constraint  (1 equation)
# ---------------------------------------------------------------------------

def relative_angle_constraint(q, body_i, body_j, n_bodies, alpha=0.0):
    """
    Constrain the relative angle between body_i and body_j.

    Equation (1):
        θ_i - θ_j - alpha = 0

    Typical use: slider constrained to slide in the direction of another body
    (e.g., θ_slider = θ_guide + 0).

    Parameters
    ----------
    q       : (3*n_bodies,) coordinate vector
    body_i  : int, 1-based index
    body_j  : int, 1-based index
    n_bodies: int
    alpha   : float, prescribed relative angle (default 0)

    Returns
    -------
    F   : (1,)               residual vector
    Jac : (1, 3*n_bodies)   Jacobian rows
    """
    n = 3 * n_bodies
    ii = _idx(body_i)
    ij = _idx(body_j)

    F = np.array([q[ii+2] - q[ij+2] - alpha])

    Jac = np.zeros((1, n))
    Jac[0, ii+2] =  1.0
    Jac[0, ij+2] = -1.0

    return F, Jac


# ---------------------------------------------------------------------------
# Horizontal prismatic slot constraint  (2 equations)
# ---------------------------------------------------------------------------

def prismatic_horizontal_constraint(q, body_i, n_bodies, y_offset=0.0, theta_fixed=0.0):
    """
    Constrain body_i to slide along a horizontal slot.

    Equations (2):
        Ry_i - y_offset  = 0
        θ_i  - theta_fixed = 0

    Parameters
    ----------
    q           : (3*n_bodies,) coordinate vector
    body_i      : int, 1-based body index
    n_bodies    : int
    y_offset    : float, fixed y position of the slider (default 0)
    theta_fixed : float, fixed orientation of the slider (default 0)

    Returns
    -------
    F   : (2,)               residual vector
    Jac : (2, 3*n_bodies)   Jacobian rows
    """
    n = 3 * n_bodies
    ix = _idx(body_i)

    F = np.array([
        q[ix+1] - y_offset,
        q[ix+2] - theta_fixed
    ])

    Jac = np.zeros((2, n))
    Jac[0, ix+1] = 1.0
    Jac[1, ix+2] = 1.0

    return F, Jac


# ---------------------------------------------------------------------------
# General prismatic joint constraint  (2 equations)
# ---------------------------------------------------------------------------

def prismatic_constraint(q, body_i, body_j, n_bodies, s_i, s_j, e_i):
    """
    General prismatic joint between body_i (slider) and body_j (guide body).

    Two constraints:
      1. Relative angle: θ_i - θ_j = 0   (they remain parallel)
      2. Transverse displacement: the slider reference point stays on the
         line defined by the guide body's axis.

         e_j^T (R_i + A(θ_i)s_i - R_j - A(θ_j)s_j) = 0

         where e_j = A(θ_j) e_i  is the unit normal to the slide direction.

    Parameters
    ----------
    q       : (3*n_bodies,) coordinate vector
    body_i  : int, 1-based index of slider body
    body_j  : int, 1-based index of guide  body
    n_bodies: int
    s_i     : (2,) local coords of reference point on body_i
    s_j     : (2,) local coords of reference point on body_j
    e_i     : (2,) unit normal in body_j frame perpendicular to slide direction

    Returns
    -------
    F   : (2,)               residual vector
    Jac : (2, 3*n_bodies)   Jacobian rows
    """
    n = 3 * n_bodies
    s_i = np.asarray(s_i, dtype=float)
    s_j = np.asarray(s_j, dtype=float)
    e_i = np.asarray(e_i, dtype=float)

    ii = _idx(body_i)
    ij = _idx(body_j)
    theta_i = q[ii+2]
    theta_j = q[ij+2]

    R_i = q[ii:ii+2]
    R_j = q[ij:ij+2]

    e_j = _A(theta_j) @ e_i          # normal direction in global frame
    d   = R_i + _A(theta_i) @ s_i - R_j - _A(theta_j) @ s_j

    F = np.array([
        theta_i - theta_j,           # relative angle
        e_j @ d                      # transverse displacement
    ])

    Jac = np.zeros((2, n))

    # Row 0: ∂(θ_i - θ_j)/∂q
    Jac[0, ii+2] =  1.0
    Jac[0, ij+2] = -1.0

    # Row 1: ∂(e_j^T d)/∂q
    # ∂/∂R_i  = e_j^T
    Jac[1, ii]   = e_j[0]
    Jac[1, ii+1] = e_j[1]
    # ∂/∂θ_i  = e_j^T dA(θ_i) s_i
    Jac[1, ii+2] = e_j @ (_dA(theta_i) @ s_i)
    # ∂/∂R_j  = -e_j^T
    Jac[1, ij]   = -e_j[0]
    Jac[1, ij+1] = -e_j[1]
    # ∂/∂θ_j  = (dA(θ_j) e_i)^T d + e_j^T (-dA(θ_j) s_j)
    #          = (dA(θ_j) e_i)^T d - e_j^T (dA(θ_j) s_j)
    Jac[1, ij+2] = (_dA(theta_j) @ e_i) @ d - e_j @ (_dA(theta_j) @ s_j)

    return F, Jac


# ---------------------------------------------------------------------------
# Assembly helper
# ---------------------------------------------------------------------------

def assemble(constraints):
    """
    Assemble a list of constraint tuples into a single (F, Jac) pair.

    Parameters
    ----------
    constraints : list of (F_i, Jac_i) tuples
        Each element is the output of one constraint function call.

    Returns
    -------
    F   : (nc_total,)              stacked residual vector
    Jac : (nc_total, 3*n_bodies)   stacked Jacobian
    """
    F_list   = [c[0] for c in constraints]
    Jac_list = [c[1] for c in constraints]
    return np.concatenate(F_list), np.vstack(Jac_list)
