"""
utils — 2D Multibody Kinematics Toolkit
========================================
Provides reusable constraint functions and solvers for planar mechanisms.

Quick start
-----------
    from utils import (
        # Constraints
        ground_constraint,
        revolute_constraint,
        driving_angle_constraint,
        relative_angle_constraint,
        prismatic_horizontal_constraint,
        prismatic_constraint,
        assemble,
        # Solver
        newton_raphson,
        newton_raphson_trajectory,
        # Kinematics
        velocity_analysis,
        acceleration_analysis,
        build_b_velocity,
        build_b_acceleration,
        nullspace_basis,
    )
"""

from .constraints_2d import (
    ground_constraint,
    revolute_constraint,
    driving_angle_constraint,
    relative_angle_constraint,
    prismatic_horizontal_constraint,
    prismatic_constraint,
    assemble,
)

from .solver import (
    newton_raphson,
    newton_raphson_trajectory,
)

from .kinematics import (
    velocity_analysis,
    acceleration_analysis,
    build_b_velocity,
    build_b_acceleration,
    nullspace_basis,
)

__all__ = [
    # constraints
    "ground_constraint",
    "revolute_constraint",
    "driving_angle_constraint",
    "relative_angle_constraint",
    "prismatic_horizontal_constraint",
    "prismatic_constraint",
    "assemble",
    # solver
    "newton_raphson",
    "newton_raphson_trajectory",
    # kinematics
    "velocity_analysis",
    "acceleration_analysis",
    "build_b_velocity",
    "build_b_acceleration",
    "nullspace_basis",
]
