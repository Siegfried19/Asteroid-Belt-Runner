"""Static nozzle map: pin the F8C in place and light up all 17 thruster nozzles at once so
you can see exactly where each one sits and which way it fires.

The ship is fixed (physics is never integrated -- only mj_forward to place the nozzle sites),
the camera slowly orbits. Flames are colored BY GROUP (orange=main, red=reverse, cyan=RCS) and
labeled with the thruster name (toggle labels in the viewer if hidden). A legend table prints
to the console. Use --cycle to instead fire one nozzle at a time, naming each as it lights.

Run from repo root (needs a display):
    conda run -n asteroid-belt-runner python Agent_tool/thruster_layout_viewer.py
    conda run -n asteroid-belt-runner python Agent_tool/thruster_layout_viewer.py --cycle
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import mujoco.viewer
import numpy as np

from envs.belt_generator import BeltConfig, build_scene
from envs.thruster_layout import THRUSTERS, THRUSTER_NAMES
from Agent_tool.thruster_flames import add_flame_geoms, intensities_from


def _print_legend():
    print(f"\n  {len(THRUSTERS)} thrusters (body frame: +X nose, +Y left, +Z up)")
    print(f"  {'name':<18} {'group':<8} {'pos (m)':<22} {'dir':<12}")
    for name, pos, d, _thrust, group in THRUSTERS:
        print(f"  {name:<18} {group:<8} {str(tuple(pos)):<22} {str(tuple(d)):<12}")
    print("  colors: main=orange  reverse=red  rcs=cyan\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cycle", action="store_true", help="fire one nozzle at a time (else all at once)")
    p.add_argument("--dwell", type=float, default=1.2, help="seconds per nozzle in --cycle mode")
    p.add_argument("--intensity", type=float, default=0.7, help="flame size for the static map [0,1]")
    args = p.parse_args()

    model, _spec, _ast = build_scene(BeltConfig(n_asteroids=0), dynamics="realistic")
    data = mujoco.MjData(model)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spacecraft")
    qadr = model.jnt_qposadr[model.body_jntadr[bid]]
    data.qpos[qadr:qadr + 7] = [0, 0, 0, 1, 0, 0, 0]
    mujoco.mj_forward(model, data)          # place the nozzle sites; never integrate (ship pinned)

    _print_legend()
    all_on = np.full(len(THRUSTER_NAMES), args.intensity)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 50.0
        viewer.cam.elevation = -18.0
        i = 0
        last_switch = 0.0
        sim_t = 0.0
        while viewer.is_running():
            if args.cycle:
                if sim_t - last_switch >= args.dwell:
                    i = (i + 1) % len(THRUSTERS)
                    last_switch = sim_t
                    name, pos, d, _t, group = THRUSTERS[i]
                    print(f"  -> {name}  ({group})  pos={tuple(pos)}  dir={tuple(d)}")
                intensity = intensities_from({THRUSTER_NAMES[i]: args.intensity})
            else:
                intensity = all_on

            viewer.user_scn.ngeom = 0
            add_flame_geoms(viewer.user_scn, model, data, intensity, color_by_group=True)
            viewer.cam.azimuth += 0.15          # slow orbit so you can read the 3D layout
            viewer.sync()
            time.sleep(0.02)
            sim_t += 0.02
    print("done.")


if __name__ == "__main__":
    main()
