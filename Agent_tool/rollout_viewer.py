"""Watch a trained PPO policy fly the belt in the passive viewer.

Run from repo root (needs a display):
    conda run -n asteroid-belt-runner python Agent_tool/rollout_viewer.py \
        --model logs/ppo_simplified/best/best_model.zip --episodes 5

The scene is compiled once (reset only re-scatters the rocks via qpos/qvel), so the
MjModel/MjData and the viewer handle stay valid across episodes.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

from envs.asteroid_belt_env import AsteroidBeltEnv
from envs.belt_generator import BeltConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="path to a trained model .zip")
    p.add_argument("--n-asteroids", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--exit-r", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                   help="override exit off-axis range to match the model's training "
                        "(e.g. --exit-r 0 0 for a straight traverse)")
    p.add_argument("--belt-len", type=float, default=None, help="far edge of the belt (m); match training")
    p.add_argument("--goal-mode", choices=["traverse", "interior_point"], default="traverse",
                   help="match the mode the model was trained on")
    p.add_argument("--arrival-speed", type=float, default=None,
                   help="interior_point: require speed <= this on arrival (match training tier)")
    args = p.parse_args()

    belt_far = args.belt_len if args.belt_len else BeltConfig().belt_x_range[1]
    cfg = BeltConfig(n_asteroids=args.n_asteroids, belt_x_range=(100.0, belt_far), seed=args.seed)
    env = AsteroidBeltEnv(cfg=cfg, randomize_belt=False,
                          goal_mode=args.goal_mode, arrival_speed=args.arrival_speed)
    if args.exit_r is not None:
        env.set_exit_r(args.exit_r[0], args.exit_r[1])
    model = PPO.load(args.model, device="cpu")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.distance = 200.0
        for ep in range(args.episodes):
            obs, _ = env.reset()
            done, ret = False, 0.0
            while not done and viewer.is_running():
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(action)
                ret += r
                done = term or trunc
                viewer.cam.lookat[:] = env.ship_pos
                viewer.sync()
                time.sleep(env.model.opt.timestep * env.frame_skip)
            print(f"ep {ep}: return={ret:.1f} outcome={info.get('outcome')}")
            if not viewer.is_running():
                break
    env.close()


if __name__ == "__main__":
    main()
