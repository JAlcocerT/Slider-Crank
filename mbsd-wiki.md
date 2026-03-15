# Multibody System Dynamics — Wiki

A practical guide to the 2D kinematics toolkit in `utils/` and the Slider-Crank notebooks.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Project layout](#2-project-layout)
3. [Core concepts](#3-core-concepts)
4. [Running the example script](#4-running-the-example-script)
5. [Using the utils in your own mechanism](#5-using-the-utils-in-your-own-mechanism)
6. [Running the Jupyter notebooks](#6-running-the-jupyter-notebooks)
7. [Running the Streamlit app](#7-running-the-streamlit-app)
8. [Constraint reference](#8-constraint-reference)

---

## 1. Prerequisites

```bash
pip install numpy scipy sympy plotly streamlit
```

Or use the pinned versions from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Only `numpy` is required for the `utils/` package. The notebooks additionally use `sympy` and `scipy`. The Streamlit app needs `plotly` and `streamlit`.

---

## 2. Project layout

```
Slider-Crank/
│
├── utils/                          ← reusable kinematics toolkit
│   ├── __init__.py                 ← package entry-point (import from here)
│   ├── constraints_2d.py           ← constraint functions + analytical Jacobians
│   ├── solver.py                   ← Newton-Raphson position solver
│   ├── kinematics.py               ← velocity, acceleration, nullspace
│   └── example_slider_crank.py     ← worked example (run this to verify setup)
│
├── Python_Symbolic.ipynb           ← original SymPy slider-crank notebook
├── Python_Symbolic_Jacobian_PR.ipynb ← Jacobian derivation + four-bar
├── Slider_Crank-Position.ipynb     ← position-only analysis
├── streamlit_app.py                ← interactive animation app
└── requirements.txt
```

---

## 3. Core concepts

### Coordinate vector

Every rigid body is described by **3 coordinates**: `(Rx, Ry, θ)` — the position of its reference point in the global frame plus its orientation angle.

For a mechanism with `n` bodies the full coordinate vector is:

```
q = [Rx_1, Ry_1, θ_1,  Rx_2, Ry_2, θ_2,  ...,  Rx_n, Ry_n, θ_n]
```

Body `i` (1-based) occupies indices `[3*(i-1), 3*(i-1)+1, 3*(i-1)+2]`.

### Rotation matrix

```
A(θ) = [[cos θ,  -sin θ],
         [sin θ,   cos θ]]

dA/dθ  = [[-sin θ, -cos θ],
           [ cos θ, -sin θ]]
```

### Constraint equations

Each joint type produces algebraic equations `F(q) = 0`. The Jacobian `Fq = ∂F/∂q` is used for:

| Problem | Linear system |
|---|---|
| Position | Newton-Raphson: `Fq Δq = −F` |
| Velocity | `Fq q̇ = b_v` |
| Acceleration | `Fq q̈ = b_a − γ` |

### Degrees of freedom (mobility)

```
DOF = 3·n_bodies − n_constraints
```

A correctly assembled mechanism should have DOF = number of prescribed drivers.

---

## 4. Running the example script

From the `Slider-Crank/` directory:

```bash
python3 utils/example_slider_crank.py
```

Expected output:

```
=== Slider-Crank: single position solve at θ=45° ===
  iter   0  ||F||_inf = 7.854e-01
  iter   1  ||F||_inf = 4.393e-02
  iter   2  ||F||_inf = 1.869e-04
  iter   3  ||F||_inf = 1.728e-09
  iter   4  ||F||_inf = 6.939e-18
  Slider x = 0.451546 m
  Rod angle = -9.2178°

=== Nullspace (mobility check) ===
  Jacobian shape: (12, 12)
  Nullspace basis shape: (12, 0)  (columns = DOF)

=== Sweeping crank 0→360°, ω=10 rad/s ===
  Slider x range: [0.1937, 0.4975] m
  Slider vx range: [-1.5582, 1.7570] m/s
  Slider ax range: [-21.6305, 13.0289] m/s²
Done.
```

The nullspace having 0 columns confirms the mechanism is fully determined (1 DOF prescribed by the driving angle constraint).

---

## 5. Using the utils in your own mechanism

### Step 1 — import

```python
from utils import (
    ground_constraint,
    revolute_constraint,
    driving_angle_constraint,
    prismatic_horizontal_constraint,
    assemble,
    newton_raphson,
    velocity_analysis,
    acceleration_analysis,
    build_b_velocity,
    build_b_acceleration,
)
import numpy as np
```

### Step 2 — define your constraint function

Write one function that assembles all constraints for a given parameter (e.g., driven crank angle):

```python
N_BODIES = 4
L2, L3, OFFSET = 0.15, 0.35, 0.05

def my_constraints(q, theta_driven):
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
```

### Step 3 — position solve

```python
q0 = np.array([0, 0, 0,  0, 0, 0,  L2, 0, 0,  L2+L3, OFFSET, 0], dtype=float)
theta = np.deg2rad(45)

fn = lambda q: my_constraints(q, theta)
q_sol, converged = newton_raphson(q0, fn, verbose=True)

print(f"Slider x = {q_sol[9]:.4f} m")   # index = 3*(4-1)+0 = 9
```

### Step 4 — velocity analysis

```python
F, Jac = fn(q_sol)
nc = len(F)

omega = 10.0  # rad/s (crank angular velocity)
driven_row = nc - 1  # driving-angle constraint is the last row

b_v  = build_b_velocity(nc, [driven_row], [omega])
qdot = velocity_analysis(Jac, b_v)

print(f"Slider vx = {qdot[9]:.4f} m/s")
```

### Step 5 — acceleration analysis

```python
alpha = 0.0  # rad/s² (crank angular acceleration)

b_a      = build_b_acceleration(nc, [driven_row], [alpha])
qdotdot  = acceleration_analysis(q_sol, qdot, Jac, fn, b_a)

print(f"Slider ax = {qdotdot[9]:.4f} m/s²")
```

### Angle sweep with warm-starting

```python
from utils import newton_raphson_trajectory

angles = np.linspace(0, 2*np.pi, 361)

# factory: returns a constraint function for each driven angle
factory = lambda theta: (lambda q: my_constraints(q, theta))

q_traj, converged = newton_raphson_trajectory(q0, factory, angles)
x_slider = q_traj[:, 9]   # Rx of body 4 at each step
```

---

## 6. Running the Jupyter notebooks

```bash
jupyter notebook
```

Then open one of:

| Notebook | What it shows |
|---|---|
| `Slider_Crank-Position.ipynb` | Position analysis with SymPy + fsolve |
| `Python_Symbolic.ipynb` | Full slider-crank with SymPy constraints + Dash animation |
| `Python_Symbolic_Jacobian_PR.ipynb` | Analytical Jacobian derivation; four-bar mechanism |
| `Python_4B_Example.ipynb` | Four-bar linkage example |

> **Note:** The SymPy notebooks rebuild symbolic expressions on every call to the solver, which is slow for parameter sweeps. The `utils/` package avoids this by using pure numpy.

---

## 7. Running the Streamlit app

```bash
streamlit run streamlit_app.py
```

This opens an interactive browser app to visualise the slider-crank motion. Adjust link lengths and crank speed with the sidebar sliders.

---

## 8. Constraint reference

All functions live in `utils/constraints_2d.py` and return `(F, Jac)`.

### `ground_constraint(q, body_i, n_bodies, Rx0=0, Ry0=0, theta0=0)`

Fixes a body to the global frame. Produces **3 equations**.

```
Rx_i = Rx0,  Ry_i = Ry0,  θ_i = theta0
```

### `revolute_constraint(q, body_i, body_j, n_bodies, s_i, s_j)`

Pin joint between two bodies. Produces **2 equations**.

```
R_i + A(θ_i) s_i − R_j − A(θ_j) s_j = 0
```

- `s_i`, `s_j`: local-frame coordinates of the joint point on each body (e.g., `[L, 0]` for the far end of a link of length L).

### `driving_angle_constraint(q, body_i, n_bodies, theta_driven)`

Prescribes the absolute orientation of a body. Produces **1 equation**.

```
θ_i = theta_driven
```

### `relative_angle_constraint(q, body_i, body_j, n_bodies, alpha=0)`

Constrains the relative angle between two bodies. Produces **1 equation**.

```
θ_i − θ_j = alpha
```

### `prismatic_horizontal_constraint(q, body_i, n_bodies, y_offset=0, theta_fixed=0)`

Constrains a body to slide along a horizontal line. Produces **2 equations**.

```
Ry_i = y_offset,  θ_i = theta_fixed
```

### `prismatic_constraint(q, body_i, body_j, n_bodies, s_i, s_j, e_i)`

General prismatic joint between a slider body and a guide body. Produces **2 equations**: one for relative angle, one for transverse displacement.

### `assemble(list_of_tuples)`

Stacks a list of `(F_i, Jac_i)` tuples into a single `(F, Jac)` pair ready for the solver.

```python
F, Jac = assemble([
    ground_constraint(...),
    revolute_constraint(...),
    ...
])
```
