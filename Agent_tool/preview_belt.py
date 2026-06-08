"""Eyeball a procedurally generated asteroid belt in the MuJoCo passive viewer.

Run from repo root (needs a display):
    conda run -n space-robotics-project python Agent_tool/preview_belt.py --n 80 --seed 3
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import mujoco.viewer

from envs.belt_generator import BeltConfig, build_scene


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=60, help="number of asteroids")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = BeltConfig(n_asteroids=args.n, seed=args.seed)
    model, _, info = build_scene(cfg)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"Belt: {len(info)} asteroids | ngeom={model.ngeom}. Close the window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [200, 0, 0]
        viewer.cam.distance = 500.0
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
