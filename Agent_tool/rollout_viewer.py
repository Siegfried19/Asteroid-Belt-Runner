"""Watch a trained PPO policy fly the belt in the passive viewer.

Run from repo root (needs a display):
    conda run -n space-robotics-project python Agent_tool/rollout_viewer.py \
        --model logs/ppo_simplified/best/best_model.zip --episodes 5

Uses randomize_belt=False so the MjModel/MjData persist across resets and the
viewer handle stays valid.
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
    args = p.parse_args()

    cfg = BeltConfig(n_asteroids=args.n_asteroids, seed=args.seed)
    env = AsteroidBeltEnv(cfg=cfg, randomize_belt=False)
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
