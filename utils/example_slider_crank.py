"""
Slider-Crank Example using the utils package
============================================
Demonstrates how to use the reusable constraint functions to set up
and solve the slider-crank position, velocity, and acceleration problems.

Mechanism layout (4 bodies, reference-point coordinates):
    Body 1: Ground frame  (fixed)
    Body 2: Crank         (rotates about O at body-1 origin)
    Body 3: Connecting rod
    Body 4: Slider        (translates along x-axis)

Constraints:
    ground(1)        : Rx1=0, Ry1=0, θ1=0                  (3 eqs)
    revolute(1,2)    : pin at O, s_i=(0,0), s_j=(0,0)       (2 eqs)
    revolute(2,3)    : pin at A, s_i=(L2,0), s_j=(0,0)      (2 eqs)
    revolute(3,4)    : pin at B, s_i=(L3,0), s_j=(0,0)      (2 eqs)
    prismatic_h(4)   : Ry4=offset, θ4=0                     (2 eqs)
    driving_angle(2) : θ2=θ_driven                          (1 eq)
                                                      Total: 12 eqs = 4*3 DOF ✓
"""

import numpy as np
import sys
import os

# Allow running from the Slider-Crank directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import (
    ground_constraint,
    revolute_constraint,
    driving_angle_constraint,
    prismatic_horizontal_constraint,
    assemble,
    newton_raphson,
    newton_raphson_trajectory,
    velocity_analysis,
    acceleration_analysis,
    build_b_velocity,
    build_b_acceleration,
    nullspace_basis,
)

# ---------------------------------------------------------------------------
# Mechanism parameters
# ---------------------------------------------------------------------------
L2     = 0.15    # crank length  [m]
L3     = 0.35    # rod length    [m]
OFFSET = 0.05    # slider y-offset from crank pivot [m]
N_BODIES = 4


def slider_crank_constraints(q, theta_driven):
    """Return (F, Jac) for the slider-crank at crank angle theta_driven."""
    return assemble([
        ground_constraint(q, body_i=1, n_bodies=N_BODIES),
        revolute_constraint(q, body_i=1, body_j=2, n_bodies=N_BODIES,
                            s_i=[0, 0], s_j=[0, 0]),
        revolute_constraint(q, body_i=2, body_j=3, n_bodies=N_BODIES,
                            s_i=[L2, 0], s_j=[0, 0]),
        revolute_constraint(q, body_i=3, body_j=4, n_bodies=N_BODIES,
                            s_i=[L3, 0], s_j=[0, 0]),
        prismatic_horizontal_constraint(q, body_i=4, n_bodies=N_BODIES,
                                        y_offset=OFFSET, theta_fixed=0.0),
        driving_angle_constraint(q, body_i=2, n_bodies=N_BODIES,
                                 theta_driven=theta_driven),
    ])


# ---------------------------------------------------------------------------
# Initial guess (bodies roughly in line, crank at 0 deg)
# ---------------------------------------------------------------------------
# q = [Rx1, Ry1, θ1,  Rx2, Ry2, θ2,  Rx3, Ry3, θ3,  Rx4, Ry4, θ4]
q0 = np.array([
    0.0,  0.0,  0.0,   # body 1 (ground)
    0.0,  0.0,  0.0,   # body 2 (crank)   — ref point at crank pivot
    L2,   0.0,  0.0,   # body 3 (rod)     — ref point at pin A
    L2+L3, OFFSET, 0.0 # body 4 (slider)  — ref point at pin B
])


# ---------------------------------------------------------------------------
# Single position solve
# ---------------------------------------------------------------------------
def solve_position(theta_deg, q_init=None, verbose=False):
    if q_init is None:
        q_init = q0
    theta = np.deg2rad(theta_deg)
    fn = lambda q: slider_crank_constraints(q, theta)
    q, ok = newton_raphson(q_init, fn, verbose=verbose)
    if not ok:
        print(f"  WARNING: position solve did not converge at θ={theta_deg}°")
    return q


# ---------------------------------------------------------------------------
# Full kinematic sweep
# ---------------------------------------------------------------------------
def full_sweep(angles_deg=None, omega=10.0, alpha_crank=0.0):
    """
    Sweep crank through angles_deg, returning slider x-position,
    velocity, and acceleration at each step.

    Parameters
    ----------
    angles_deg : array of crank angles in degrees (default: 0 to 360, 361 pts)
    omega      : crank angular velocity [rad/s]
    alpha_crank: crank angular acceleration [rad/s²]
    """
    if angles_deg is None:
        angles_deg = np.linspace(0, 360, 361)

    angles_rad = np.deg2rad(angles_deg)
    n_steps    = len(angles_rad)
    n          = len(q0)
    nc         = 12  # total constraint equations

    q_traj    = np.zeros((n_steps, n))
    x_slider  = np.zeros(n_steps)
    vx_slider = np.zeros(n_steps)
    ax_slider = np.zeros(n_steps)

    # driving angle is constraint row 11 (0-based), last equation
    driven_row = 11

    q = q0.copy()
    for k, theta in enumerate(angles_rad):
        fn = lambda q, t=theta: slider_crank_constraints(q, t)

        # Position
        q, ok = newton_raphson(q, fn)
        if not ok:
            print(f"  WARNING: did not converge at step {k}, θ={angles_deg[k]:.1f}°")
        q_traj[k] = q
        x_slider[k] = q[9]  # Rx of body 4 = index 3*(4-1)+0 = 9

        # Velocity (b_v: only driven-angle row is non-zero = omega)
        _, Jac = fn(q)
        b_v = build_b_velocity(nc, [driven_row], [omega])
        qdot = velocity_analysis(Jac, b_v)
        vx_slider[k] = qdot[9]

        # Acceleration (b_a: only driven-angle row is non-zero = alpha_crank)
        b_a = build_b_acceleration(nc, [driven_row], [alpha_crank])
        qdotdot = acceleration_analysis(q, qdot, Jac, fn, b_a)
        ax_slider[k] = qdotdot[9]

    return angles_deg, x_slider, vx_slider, ax_slider


# ---------------------------------------------------------------------------
# Run a quick demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Slider-Crank: single position solve at θ=45° ===")
    q_sol = solve_position(45.0, verbose=True)
    print(f"  Slider x = {q_sol[9]:.6f} m")
    print(f"  Rod angle = {np.rad2deg(q_sol[8]):.4f}°")

    print()
    print("=== Nullspace (mobility check) ===")
    theta = np.deg2rad(45)
    _, Jac = slider_crank_constraints(q_sol, theta)
    R = nullspace_basis(Jac)
    print(f"  Jacobian shape: {Jac.shape}")
    print(f"  Nullspace basis shape: {R.shape}  (columns = DOF)")

    print()
    print("=== Sweeping crank 0→360°, ω=10 rad/s ===")
    angles, xs, vxs, axs = full_sweep(np.linspace(0, 360, 73), omega=10.0)
    print(f"  Slider x range: [{xs.min():.4f}, {xs.max():.4f}] m")
    print(f"  Slider vx range: [{vxs.min():.4f}, {vxs.max():.4f}] m/s")
    print(f"  Slider ax range: [{axs.min():.4f}, {axs.max():.4f}] m/s²")
    print("Done.")
