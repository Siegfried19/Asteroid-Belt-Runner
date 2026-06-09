"""Demo: watch the F8C's 17 fixed-direction thrusters fire, flames sized by thrust.

The ship runs a short "airshow" of clean, symmetric thruster firings (forward burn, reverse
brake, strafe up/down, yaw left/right). Each maneuver breathes its thrust 0 -> max -> 0 so you
can see the flame plume grow and shrink with thrust magnitude. The nozzle directions are FIXED
(no gimballing) -- the flames show exactly where and which way each thruster pushes.

Run from repo root (needs a display):
    conda run -n asteroid-belt-runner python Agent_tool/thruster_flame_viewer.py
"""
import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import mujoco.viewer
import numpy as np

from envs.belt_generator import BeltConfig, build_scene
from envs.thruster_layout import THRUSTER_NAMES, MAX_THRUSTS
from Agent_tool.thruster_flames import add_flame_geoms, intensities_from

# Clean, pure maneuvers built from the symmetric thruster groups (no net tumble).
MANEUVERS = [
    ("MAIN BURN   (accelerate +X)", ["main_l", "main_r"]),
    ("REVERSE     (brake -X)",      ["rev_c", "rev_l", "rev_r"]),
    ("STRAFE UP   (+Z)",            ["rcs_fl_zp", "rcs_fr_zp", "rcs_al_zp", "rcs_ar_zp"]),
    ("STRAFE DOWN (-Z)",            ["rcs_fl_zn", "rcs_fr_zn", "rcs_al_zn", "rcs_ar_zn"]),
    ("STRAFE LEFT (+Y)",            ["rcs_f_yp", "rcs_a_yp"]),
    ("STRAFE RIGHT(-Y)",            ["rcs_f_yn", "rcs_a_yn"]),
    ("PITCH UP   (+My)",            ["rcs_fl_zp", "rcs_fr_zp", "rcs_al_zn", "rcs_ar_zn"]),
    ("ROLL       (+Mx)",            ["rcs_fl_zp", "rcs_al_zp", "rcs_fr_zn", "rcs_ar_zn"]),
    ("YAW        (+Mz)",            ["rcs_f_yp", "rcs_a_yn"]),
]


def _ship_qadr(model):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spacecraft")
    return model.jnt_qposadr[model.body_jntadr[bid]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hold", type=float, default=3.0, help="seconds per maneuver")
    p.add_argument("--loops", type=int, default=3, help="times to cycle through all maneuvers")
    args = p.parse_args()

    model, _spec, _ast = build_scene(BeltConfig(n_asteroids=0), dynamics="realistic")
    data = mujoco.MjData(model)
    qadr = _ship_qadr(model)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spacecraft")
    act_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                        for n in THRUSTER_NAMES])

    def reset_pose():
        mujoco.mj_resetData(model, data)
        data.qpos[qadr:qadr + 7] = [0, 0, 0, 1, 0, 0, 0]
        mujoco.mj_forward(model, data)

    reset_pose()
    steps_per = int(args.hold / model.opt.timestep)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 55.0
        viewer.cam.elevation = -20.0
        for _loop in range(args.loops):
            for label, names in MANEUVERS:
                if not viewer.is_running():
                    break
                print(f"  {label}")
                reset_pose()
                for k in range(steps_per):
                    if not viewer.is_running():
                        break
                    # breathe thrust 0 -> 1 -> 0 over the maneuver so the flame scales visibly
                    f = math.sin(math.pi * k / steps_per)
                    intensity = intensities_from({n: f for n in names})
                    data.ctrl[act_ids] = intensity * MAX_THRUSTS
                    mujoco.mj_step(model, data)

                    viewer.user_scn.ngeom = 0
                    add_flame_geoms(viewer.user_scn, model, data, intensity)
                    viewer.cam.lookat[:] = data.xpos[bid]
                    viewer.cam.azimuth += 0.05      # slow orbit for depth
                    viewer.sync()
                    time.sleep(model.opt.timestep)
            if not viewer.is_running():
                break
    print("done.")


if __name__ == "__main__":
    main()
