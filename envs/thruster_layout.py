"""Realistic thruster layout for the F8C: 2 main + 3 reverse + 12 RCS = 17 thrusters.

Each thruster is a one-directional force applied at a hull site (MuJoCo site actuator:
the force acts at the site position, so an off-COM site naturally produces torque). The
agent commands 17 thrust magnitudes in [0, max]; control allocation becomes part of what
the policy learns.

Because `MjSpec` (3.3.x) can't delete the 6 virtual force/torque actuators inherited from
`environment.xml`, the builder *adds* these 17 actuators alongside them; the realistic env
commands only the thruster actuators (by name) and leaves the virtual ones at ctrl=0.

Coordinate frame: ship body. +X = forward/nose, +Z = up, +Y = left. Hull is ~24 m long
(capsule half-length 12) and ~12 m across. Run this file to verify 6-DOF controllability.
"""
import mujoco
import numpy as np

MAIN_THRUST = 5.0e6      # N, per main engine (2 -> ~1e7 forward, ~matches old Fx range)
REVERSE_THRUST = 1.6e6   # N, per reverse thruster (weaker, as in-game)
RCS_THRUST = 1.0e6       # N, per RCS thruster

# --- RCS geometry: tweak these to reshape the layout (all positions in m, ship frame) ---
MAIN_X = -11.0   # main engines longitudinal station (rear)
REV_X  =  11.0   # reverse thrusters longitudinal station (nose)
RCS_X  =   6.0   # fore/aft RCS station |x| (pulled in toward center -> sits on the hull)
RCS_ZY =   4.5   # vertical (±Z) quad half-spread in y -> roll moment arm ("opened" wider)

# (name, position [m], thrust direction [unit], max thrust [N], group)
# Naming: f/a = fore(+x)/aft(-x), l/r = left(+y)/right(-y); zp/zn = +Z/-Z, yp/yn = +Y/-Y.
THRUSTERS = [
    # --- 2 main engines: rear, push +X ---
    ("main_l", (MAIN_X,  2.0,  0.0), (1, 0, 0), MAIN_THRUST, "main"),
    ("main_r", (MAIN_X, -2.0,  0.0), (1, 0, 0), MAIN_THRUST, "main"),
    # --- 3 reverse thrusters: nose, push -X (braking) ---
    ("rev_c", (REV_X,  0.0,  0.0), (-1, 0, 0), REVERSE_THRUST, "reverse"),
    ("rev_l", (REV_X,  2.0,  0.0), (-1, 0, 0), REVERSE_THRUST, "reverse"),
    ("rev_r", (REV_X, -2.0,  0.0), (-1, 0, 0), REVERSE_THRUST, "reverse"),
    # --- 12 RCS: a SYMMETRIC layout (vertical quad + horizontal pair) ---
    # 8 vertical (±Z) at the 4 corners (±RCS_X, ±RCS_ZY, 0): all +Z = pure heave; fore/aft diff
    # = pitch; left/right diff = roll. Mirror-symmetric, so pure translation needs no roll cancel.
    ("rcs_fl_zp", ( RCS_X,  RCS_ZY, 0.0), (0, 0,  1), RCS_THRUST, "rcs"),
    ("rcs_fl_zn", ( RCS_X,  RCS_ZY, 0.0), (0, 0, -1), RCS_THRUST, "rcs"),
    ("rcs_fr_zp", ( RCS_X, -RCS_ZY, 0.0), (0, 0,  1), RCS_THRUST, "rcs"),
    ("rcs_fr_zn", ( RCS_X, -RCS_ZY, 0.0), (0, 0, -1), RCS_THRUST, "rcs"),
    ("rcs_al_zp", (-RCS_X+1,  RCS_ZY+1, 0.0), (0, 0,  1), RCS_THRUST, "rcs"),
    ("rcs_al_zn", (-RCS_X+1,  RCS_ZY+1, 0.0), (0, 0, -1), RCS_THRUST, "rcs"),
    ("rcs_ar_zp", (-RCS_X+1, -RCS_ZY-1, 0.0), (0, 0,  1), RCS_THRUST, "rcs"),
    ("rcs_ar_zn", (-RCS_X+1, -RCS_ZY-1, 0.0), (0, 0, -1), RCS_THRUST, "rcs"),
    # 4 horizontal (±Y) at the fore/aft centerline (±RCS_X, 0, 0): all same = pure sway;
    # fore/aft diff = yaw. (Roll is fully covered by the vertical quad above.)
    ("rcs_f_yp", ( RCS_X, 0.0, 0.0), (0,  1, 0), RCS_THRUST, "rcs"),
    ("rcs_f_yn", ( RCS_X, 0.0, 0.0), (0, -1, 0), RCS_THRUST, "rcs"),
    ("rcs_a_yp", (-RCS_X+1, 0.0, 0.0), (0,  1, 0), RCS_THRUST, "rcs"),
    ("rcs_a_yn", (-RCS_X+1, 0.0, 0.0), (0, -1, 0), RCS_THRUST, "rcs"),
]

THRUSTER_NAMES = [t[0] for t in THRUSTERS]
MAX_THRUSTS = np.array([t[3] for t in THRUSTERS], dtype=float)


def add_thrusters(spec: mujoco.MjSpec, ship_body: str = "spacecraft"):
    """Add 17 thruster sites + site actuators to the ship. Returns actuator names (in order)."""
    ship = spec.body(ship_body)
    names = []
    for name, pos, direction, thrust, group in THRUSTERS:
        site = ship.add_site()
        site.name = f"{name}_site"
        site.pos = list(pos)
        site.size = [0.3, 0, 0]
        site.rgba = [1.0, 0.5, 0.0, 1.0] if group == "main" else [0.3, 0.6, 1.0, 1.0]

        act = spec.add_actuator()
        act.name = name
        act.trntype = mujoco.mjtTrn.mjTRN_SITE
        act.target = f"{name}_site"
        d = np.asarray(direction, dtype=float)
        d = d / (np.linalg.norm(d) + 1e-12)
        act.gear = [d[0], d[1], d[2], 0, 0, 0]
        act.ctrllimited = 1
        act.ctrlrange = [0.0, float(thrust)]
        names.append(name)
    return names


def wrench_matrix():
    """6xN wrench matrix (force; torque about ship COM at origin) at unit thrust."""
    cols = []
    for _, pos, direction, _, _ in THRUSTERS:
        p = np.asarray(pos, float)
        d = np.asarray(direction, float)
        d = d / (np.linalg.norm(d) + 1e-12)
        cols.append(np.concatenate([d, np.cross(p, d)]))
    return np.array(cols).T  # 6 x N


def _controllability_report():
    W = wrench_matrix()
    rank = np.linalg.matrix_rank(W)
    axes = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    print(f"thrusters={W.shape[1]} wrench rank={rank} (need 6)")
    ok = rank == 6
    for i, ax in enumerate(axes):
        row = W[i]
        pos_ok = (row > 1e-9).any()
        neg_ok = (row < -1e-9).any()
        ok = ok and pos_ok and neg_ok
        print(f"  {ax}: +dir={'Y' if pos_ok else 'N'}  -dir={'Y' if neg_ok else 'N'}")
    print("CONTROLLABLE (rank 6, both signs each axis):", ok)
    return ok


if __name__ == "__main__":
    _controllability_report()
