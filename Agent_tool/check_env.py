"""Sanity-check the AsteroidBeltEnv: Gymnasium API compliance + random-action rollout.

Run from repo root:
    conda run -n asteroid-belt-runner python Agent_tool/check_env.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from gymnasium.utils.env_checker import check_env

from envs.asteroid_belt_env import AsteroidBeltEnv
from envs.belt_generator import BeltConfig


def check_one(dynamics):
    print(f"\n=== dynamics={dynamics} ===")
    cfg = BeltConfig(n_asteroids=40, seed=1)
    env = AsteroidBeltEnv(cfg=cfg, dynamics=dynamics, max_steps=300)
    print(f"obs space: {env.observation_space.shape}  act space: {env.action_space.shape}")

    # Gymnasium's own conformance checker (skip render to avoid needing a display)
    check_env(env.unwrapped, skip_render_check=True)
    print("[check_env] passed.")

    # random rollout, tally outcomes
    outcomes = {}
    n_ep = 20
    for ep in range(n_ep):
        obs, _ = env.reset(seed=ep)
        assert np.isfinite(obs).all(), "non-finite obs at reset"
        ep_ret, done = 0.0, False
        while not done:
            a = env.action_space.sample()
            obs, r, term, trunc, info = env.step(a)
            assert np.isfinite(obs).all(), "non-finite obs in step"
            ep_ret += r
            done = term or trunc
        outcomes[info.get("outcome", "?")] = outcomes.get(info.get("outcome", "?"), 0) + 1
    print(f"[rollout] {n_ep} random episodes, outcomes: {outcomes}")
    env.close()


def main():
    check_one("simplified")
    check_one("realistic")
    print("\nALL OK")


if __name__ == "__main__":
    main()
