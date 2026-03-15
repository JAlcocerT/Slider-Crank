"""
Slider-Crank Animator
=====================
Dark-themed matplotlib animation showing:
  - Left panel : mechanism in motion (crank, rod, slider, joint dots, trails)
  - Right panels: live x-position, velocity, acceleration plots with a
                  running cursor

Run from the Slider-Crank/ directory:
    python3 animate.py

Optional CLI args:
    python3 animate.py --omega 15 --fps 60 --save slider_crank.gif
"""

import argparse
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

# ── utils import ────────────────────────────────────────────────────────────
from utils import (
    ground_constraint, revolute_constraint, driving_angle_constraint,
    prismatic_horizontal_constraint, assemble,
    newton_raphson, velocity_analysis, acceleration_analysis,
    build_b_velocity, build_b_acceleration,
)

# ── Mechanism parameters ─────────────────────────────────────────────────────
L2      = 0.15    # crank length  [m]
L3      = 0.35    # connecting rod length [m]
OFFSET  = 0.05    # slider y-offset [m]
N_BODIES = 4
NC       = 12     # total constraint equations
DRIVEN_ROW = 11   # index of driving-angle constraint row

q0 = np.array([
    0.0,  0.0,  0.0,
    0.0,  0.0,  0.0,
    L2,   0.0,  0.0,
    L2+L3, OFFSET, 0.0
])


# ── Constraint function ───────────────────────────────────────────────────────
def constraints(q, theta):
    return assemble([
        ground_constraint(q, 1, N_BODIES),
        revolute_constraint(q, 1, 2, N_BODIES, [0,0], [0,0]),
        revolute_constraint(q, 2, 3, N_BODIES, [L2,0], [0,0]),
        revolute_constraint(q, 3, 4, N_BODIES, [L3,0], [0,0]),
        prismatic_horizontal_constraint(q, 4, N_BODIES, OFFSET, 0.0),
        driving_angle_constraint(q, 2, N_BODIES, theta),
    ])


# ── Pre-compute full kinematic sweep ─────────────────────────────────────────
def compute_sweep(n_frames=360, omega=10.0):
    angles = np.linspace(0, 2*np.pi, n_frames, endpoint=False)
    q_traj   = np.zeros((n_frames, 12))
    x_traj   = np.zeros(n_frames)
    vx_traj  = np.zeros(n_frames)
    ax_traj  = np.zeros(n_frames)

    q = q0.copy()
    for k, theta in enumerate(angles):
        fn = lambda q, t=theta: constraints(q, t)
        q, _ = newton_raphson(q, fn)
        q_traj[k] = q

        _, Jac = fn(q)
        qdot = velocity_analysis(Jac, build_b_velocity(NC, [DRIVEN_ROW], [omega]))
        qdotdot = acceleration_analysis(q, qdot, Jac, fn,
                                        build_b_acceleration(NC, [DRIVEN_ROW], [0.0]))
        x_traj[k]  = q[9]
        vx_traj[k] = qdot[9]
        ax_traj[k] = qdotdot[9]

    return np.rad2deg(angles), q_traj, x_traj, vx_traj, ax_traj


# ── Extract joint positions from q ───────────────────────────────────────────
def joint_positions(q):
    A = lambda th: np.array([[np.cos(th), -np.sin(th)],
                              [np.sin(th),  np.cos(th)]])
    # O  : crank pivot (fixed, origin)
    O  = np.array([0.0, 0.0])
    # A  : crank-rod pin  = R_2 + A(θ_2)·[L2,0]
    P_A = q[3:5] + A(q[5]) @ np.array([L2, 0])
    # B  : rod-slider pin = R_3 + A(θ_3)·[L3,0]
    P_B = q[6:8] + A(q[8]) @ np.array([L3, 0])
    # slider reference point
    P_S = q[9:11]
    return O, P_A, P_B, P_S


# ── Color palette ─────────────────────────────────────────────────────────────
BG      = "#0d1117"
PANEL   = "#161b22"
CRANK_C = "#58a6ff"   # blue
ROD_C   = "#3fb950"   # green
SLIDER_C= "#f78166"   # coral
JOINT_C = "#e3b341"   # amber
TRAIL_C = "#bc8cff"   # purple
GRID_C  = "#21262d"
TEXT_C  = "#c9d1d9"

PLOT_COLORS = [CRANK_C, ROD_C, SLIDER_C]


def rainbow_gradient(n):
    """Return n colors cycling through hue for the crank circle."""
    return [matplotlib.colormaps["hsv"](i / n) for i in range(n)]


# ── Main ──────────────────────────────────────────────────────────────────────
def build_animation(omega=10.0, fps=30, n_frames=360):
    print(f"Pre-computing {n_frames} frames (ω={omega} rad/s)…", flush=True)
    angles_deg, q_traj, x_traj, vx_traj, ax_traj = compute_sweep(n_frames, omega)
    print("Done. Building animation…", flush=True)

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 7), facecolor=BG)
    fig.patch.set_facecolor(BG)

    # GridSpec: mechanism (left, tall) | three plots (right, stacked)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1],
                          hspace=0.55, wspace=0.35,
                          left=0.05, right=0.97, top=0.93, bottom=0.09)

    ax_mech = fig.add_subplot(gs[:, 0])          # full left column
    ax_x    = fig.add_subplot(gs[0, 1])
    ax_v    = fig.add_subplot(gs[1, 1])
    ax_a    = fig.add_subplot(gs[2, 1])

    for ax in (ax_mech, ax_x, ax_v, ax_a):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT_C, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_C)
        ax.grid(True, color=GRID_C, linewidth=0.5, linestyle="--", alpha=0.7)
        ax.title.set_color(TEXT_C)
        ax.xaxis.label.set_color(TEXT_C)
        ax.yaxis.label.set_color(TEXT_C)

    # ── Mechanism panel ───────────────────────────────────────────────────────
    pad = 0.06
    ax_mech.set_xlim(-L2 - pad, L2 + L3 + pad)
    ax_mech.set_ylim(-L2 - pad, L2 + pad + 0.04)
    ax_mech.set_aspect("equal")
    ax_mech.set_title("Slider-Crank Mechanism", fontsize=12, fontweight="bold",
                       color=TEXT_C, pad=8)
    ax_mech.set_xlabel("x  [m]", fontsize=9)
    ax_mech.set_ylabel("y  [m]", fontsize=9)

    # Slider rail (dashed line)
    rail_y = OFFSET
    ax_mech.axhline(rail_y, color="#30363d", lw=1.5, ls="--", zorder=1)
    ax_mech.annotate("rail", xy=(L2+L3+pad-0.01, rail_y+0.008),
                     color="#484f58", fontsize=7, ha="right")

    # Ground cross-hatch triangle at O
    tri = mpatches.FancyArrow(0, -0.01, 0, 0.005, width=0.025,
                               color="#30363d", zorder=2)
    ax_mech.add_patch(tri)
    ax_mech.plot([-0.025, 0.025], [-0.012, -0.012], color="#30363d", lw=2, zorder=2)

    # Crank circle (faint reference)
    crank_circle = plt.Circle((0, 0), L2, color=CRANK_C, fill=False,
                               lw=0.6, ls=":", alpha=0.25, zorder=2)
    ax_mech.add_patch(crank_circle)

    # Pre-draw slider body (rectangle)
    sw, sh = 0.06, 0.03
    slider_rect = mpatches.FancyBboxPatch(
        (0, 0), sw, sh,
        boxstyle="round,pad=0.005",
        linewidth=1.5, edgecolor=SLIDER_C, facecolor=SLIDER_C + "33",
        zorder=5
    )
    ax_mech.add_patch(slider_rect)

    # Dynamic links
    ln_crank, = ax_mech.plot([], [], color=CRANK_C, lw=4,
                              solid_capstyle="round", zorder=6)
    ln_rod,   = ax_mech.plot([], [], color=ROD_C,   lw=3,
                              solid_capstyle="round", zorder=6)

    # Joint dots
    dot_O,  = ax_mech.plot(0, 0, "o", ms=10, color=JOINT_C,
                            markeredgecolor=BG, markeredgewidth=1.5, zorder=8)
    dot_A,  = ax_mech.plot([], [], "o", ms=8,  color=JOINT_C,
                            markeredgecolor=BG, markeredgewidth=1.5, zorder=8)
    dot_B,  = ax_mech.plot([], [], "o", ms=8,  color=JOINT_C,
                            markeredgecolor=BG, markeredgewidth=1.5, zorder=8)

    # Link labels
    lbl_crank = ax_mech.text(0, 0, "L₂", color=CRANK_C, fontsize=8,
                              ha="center", va="center", zorder=9)
    lbl_rod   = ax_mech.text(0, 0, "L₃", color=ROD_C,   fontsize=8,
                              ha="center", va="center", zorder=9)

    # Crank angle arc
    arc_angles = np.linspace(0, 0, 30)
    ln_arc,  = ax_mech.plot([], [], color=CRANK_C, lw=1, alpha=0.5, zorder=4)
    theta_lbl = ax_mech.text(0.04, 0.01, "", color=CRANK_C,
                              fontsize=8, zorder=9)

    # Slider x-position text
    x_lbl = ax_mech.text(0.5, -L2-0.045, "", color=SLIDER_C,
                          fontsize=8, ha="center", zorder=9)

    # Trail of pin B (fading)
    TRAIL_LEN = 60
    trail_x = np.full(TRAIL_LEN, np.nan)
    trail_y = np.full(TRAIL_LEN, np.nan)
    trail_line, = ax_mech.plot([], [], color=TRAIL_C, lw=1.2,
                                alpha=0.55, zorder=3)

    # ── Right-side plots (static full curves + live cursor) ──────────────────
    plot_data = [
        (ax_x, x_traj  * 1000, "Slider  x",  "mm",          CRANK_C),
        (ax_v, vx_traj,        "Slider  vₓ", "m/s",         ROD_C),
        (ax_a, ax_traj,        "Slider  aₓ", "m/s²",        SLIDER_C),
    ]
    cursors = []
    for ax, data, title, unit, color in plot_data:
        ax.plot(angles_deg, data, color=color, lw=1.2, alpha=0.35)
        ax.fill_between(angles_deg, data, alpha=0.08, color=color)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_ylabel(unit, fontsize=8)
        ax.set_xlim(0, 360)
        ax.set_xticks([0, 90, 180, 270, 360])

        # Active trailing segment
        seg_line, = ax.plot([], [], color=color, lw=2.0, zorder=5)
        # Cursor dot
        cur_dot,  = ax.plot([], [], "o", ms=6, color=color,
                            markeredgecolor=BG, markeredgewidth=1, zorder=6)
        # Vertical cursor line
        cur_vline = ax.axvline(x=0, color=color, lw=0.8,
                               ls="--", alpha=0.5, zorder=4)
        cursors.append((seg_line, cur_dot, cur_vline, data))

    ax_a.set_xlabel("Crank angle  θ  [°]", fontsize=8)

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.suptitle(
        f"Slider-Crank  ·  L₂={L2*1000:.0f} mm  L₃={L3*1000:.0f} mm"
        f"  ·  ω = {omega:.0f} rad/s",
        color=TEXT_C, fontsize=11, fontweight="bold", y=0.98
    )

    # ── Animation update ──────────────────────────────────────────────────────
    def update(frame):
        k = frame % n_frames
        q = q_traj[k]
        theta_deg_k = angles_deg[k]

        O, P_A, P_B, P_S = joint_positions(q)

        # — Crank link
        ln_crank.set_data([O[0], P_A[0]], [O[1], P_A[1]])
        # — Rod link
        ln_rod.set_data([P_A[0], P_B[0]], [P_A[1], P_B[1]])
        # — Joints
        dot_A.set_data([P_A[0]], [P_A[1]])
        dot_B.set_data([P_B[0]], [P_B[1]])
        # — Slider rect (centred on pin B x, at rail y)
        slider_rect.set_x(P_S[0] - sw/2)
        slider_rect.set_y(rail_y - sh/2)

        # — Link midpoint labels
        mid_crank = (O + P_A) / 2
        lbl_crank.set_position((mid_crank[0] - 0.012, mid_crank[1] + 0.012))
        mid_rod   = (P_A + P_B) / 2
        lbl_rod.set_position((mid_rod[0], mid_rod[1] + 0.015))

        # — Crank angle arc
        theta_k = np.deg2rad(theta_deg_k)
        arc_t = np.linspace(0, theta_k, 40)
        r_arc = L2 * 0.35
        ln_arc.set_data(r_arc * np.cos(arc_t), r_arc * np.sin(arc_t))
        theta_lbl.set_position((r_arc*1.35 * np.cos(theta_k/2),
                                 r_arc*1.35 * np.sin(theta_k/2)))
        theta_lbl.set_text(f"{theta_deg_k:.0f}°")

        # — Slider x label
        x_lbl.set_position((P_S[0], -L2 - 0.042))
        x_lbl.set_text(f"x = {P_S[0]*1000:.1f} mm")

        # — Trail of pin B
        trail_x[:-1] = trail_x[1:]
        trail_y[:-1] = trail_y[1:]
        trail_x[-1]  = P_B[0]
        trail_y[-1]  = P_B[1]
        trail_line.set_data(trail_x, trail_y)

        # — Right-side plot cursors
        for seg_line, cur_dot, cur_vline, data in cursors:
            # Trailing segment (last ~90 degrees of data)
            tail = 45
            start = max(0, k - tail)
            seg_line.set_data(angles_deg[start:k+1], data[start:k+1])
            cur_dot.set_data([theta_deg_k], [data[k]])
            cur_vline.set_xdata([theta_deg_k])

        return (ln_crank, ln_rod, dot_A, dot_B, slider_rect,
                lbl_crank, lbl_rod, ln_arc, theta_lbl, x_lbl, trail_line,
                *[item for tup in cursors for item in tup[:3]])

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000/fps, blit=True)
    return fig, anim


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slider-Crank Animator")
    parser.add_argument("--omega",  type=float, default=10.0,
                        help="Crank angular velocity [rad/s]  (default: 10)")
    parser.add_argument("--fps",    type=int,   default=30,
                        help="Animation frames per second       (default: 30)")
    parser.add_argument("--frames", type=int,   default=360,
                        help="Number of frames per revolution   (default: 360)")
    parser.add_argument("--save",   type=str,   default=None,
                        help="Save to GIF file instead of showing window")
    args = parser.parse_args()

    fig, anim = build_animation(omega=args.omega, fps=args.fps,
                                n_frames=args.frames)

    if args.save:
        print(f"Saving to {args.save}…")
        writer = PillowWriter(fps=args.fps)
        anim.save(args.save, writer=writer, dpi=120)
        print("Saved.")
    else:
        plt.show()
