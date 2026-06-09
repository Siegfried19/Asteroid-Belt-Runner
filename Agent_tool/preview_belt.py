"""Eyeball a procedurally generated asteroid belt in the MuJoCo passive viewer.

Drives the belt through `AsteroidBeltEnv` so the rocks show their per-episode drift +
spin (the env writes those velocities at reset). The ship is left un-thrusted (it sits
at the origin) so you can watch the belt move. Press the window's close button to exit.

Run from repo root (needs a display):
    conda run -n asteroid-belt-runner python Agent_tool/preview_belt.py --n 80 --seed 3
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import mujoco.viewer
import numpy as np

from envs.asteroid_belt_env import AsteroidBeltEnv
from envs.belt_generator import BeltConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=None,
                   help="number of asteroids (default: BeltConfig default)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reset-every", type=float, default=8.0, help="re-scatter the belt every N seconds")
    args = p.parse_args()

    cfg = BeltConfig(seed=args.seed) if args.n is None else BeltConfig(n_asteroids=args.n, seed=args.seed)
    env = AsteroidBeltEnv(cfg=cfg, max_steps=10 ** 9)
    env.reset(seed=args.seed)
    r = env.ast_r
    print(f"Belt: {len(env.asteroids)} asteroids | ngeom={env.model.ngeom} | "
          f"r_eff min={r.min():.1f} max={r.max():.1f}. Close the window to exit.")

    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    x0, x1 = env.base_cfg.belt_x_range
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = [0.5 * (x0 + x1), 0, 0]
        viewer.cam.distance = 1.5 * (x1 - x0)
        t_last = 0.0
        while viewer.is_running():
            env.step(zero)            # no thrust: ship stays put, asteroids drift + spin
            t = env.data.time
            if t - t_last > args.reset_every:
                env.reset()
                t_last = t
            viewer.sync()
            time.sleep(env.model.opt.timestep * env.frame_skip)
    env.close()


if __name__ == "__main__":
    main()
