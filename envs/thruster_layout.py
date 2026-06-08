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

# (name, position [m], thrust direction [unit], max thrust [N], group)
THRUSTERS = [
    # --- 2 main engines: rear, push +X ---
    ("main_l", (-11.0,  2.0,  0.0), (1, 0, 0), MAIN_THRUST, "main"),
    ("main_r", (-11.0, -2.0,  0.0), (1, 0, 0), MAIN_THRUST, "main"),
    # --- 3 reverse thrusters: nose, push -X (braking) ---
    ("rev_c", (11.0,  0.0,  0.0), (-1, 0, 0), REVERSE_THRUST, "reverse"),
    ("rev_l", (11.0,  2.0,  0.0), (-1, 0, 0), REVERSE_THRUST, "reverse"),
    ("rev_r", (11.0, -2.0,  0.0), (-1, 0, 0), REVERSE_THRUST, "reverse"),
    # --- 12 RCS: fore (x=+9) and aft (x=-9) quads, ±Y and ±Z, offset for roll authority ---
    # ±Y thrusters offset in z -> produce roll; fore/aft pair -> yaw or pure Y
    ("rcs_fore_yp_zt", ( 9.0, 0.0,  3.0), (0,  1, 0), RCS_THRUST, "rcs"),
    ("rcs_fore_yn_zt", ( 9.0, 0.0,  3.0), (0, -1, 0), RCS_THRUST, "rcs"),
    ("rcs_aft_yp_zb",  (-9.0, 0.0, -3.0), (0,  1, 0), RCS_THRUST, "rcs"),
    ("rcs_aft_yn_zb",  (-9.0, 0.0, -3.0), (0, -1, 0), RCS_THRUST, "rcs"),
    # ±Z thrusters offset in y -> produce roll; fore/aft pair -> pitch or pure Z
    ("rcs_fore_zp_yr", ( 9.0,  3.0, 0.0), (0, 0,  1), RCS_THRUST, "rcs"),
    ("rcs_fore_zn_yr", ( 9.0,  3.0, 0.0), (0, 0, -1), RCS_THRUST, "rcs"),
    ("rcs_aft_zp_yl",  (-9.0, -3.0, 0.0), (0, 0,  1), RCS_THRUST, "rcs"),
    ("rcs_aft_zn_yl",  (-9.0, -3.0, 0.0), (0, 0, -1), RCS_THRUST, "rcs"),
    # roll couples (z-thrust at ±y) for dedicated roll authority both signs
    ("rcs_roll_a", (0.0,  4.0, 0.0), (0, 0,  1), RCS_THRUST, "rcs"),
    ("rcs_roll_b", (0.0, -4.0, 0.0), (0, 0, -1), RCS_THRUST, "rcs"),
    ("rcs_roll_c", (0.0,  4.0, 0.0), (0, 0, -1), RCS_THRUST, "rcs"),
    ("rcs_roll_d", (0.0, -4.0, 0.0), (0, 0,  1), RCS_THRUST, "rcs"),
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
