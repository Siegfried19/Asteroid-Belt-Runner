"""Evaluate a trained policy headlessly: success rate + outcome breakdown.

Run from repo root:
    conda run -n asteroid-belt-runner python Agent_tool/eval_policy.py \
        --model logs/ppo_simplified_v1/best/best_model.zip --episodes 100
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO

from envs.asteroid_belt_env import AsteroidBeltEnv
from envs.belt_generator import BeltConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dynamics", choices=["simplified", "realistic"], default="simplified")
    p.add_argument("--n-asteroids", type=int, default=135)
    p.add_argument("--max-steps", type=int, default=2200)
    p.add_argument("--exit-r", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                   help="override exit off-axis range for eval (default: env default)")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=5000)
    args = p.parse_args()

    cfg = BeltConfig(n_asteroids=args.n_asteroids, seed=args.seed)
    env = AsteroidBeltEnv(cfg=cfg, dynamics=args.dynamics, max_steps=args.max_steps,
                          randomize_belt=True)
    if args.exit_r is not None:
        env.set_exit_r(args.exit_r[0], args.exit_r[1])
    model = PPO.load(args.model, device="cpu")

    outcomes, returns = {}, []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done, ret = False, 0.0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            ret += r
            done = term or trunc
        o = info.get("outcome", "?")
        outcomes[o] = outcomes.get(o, 0) + 1
        returns.append(ret)
    env.close()

    n = args.episodes
    print(f"model: {args.model}  dynamics: {args.dynamics}  episodes: {n}")
    print(f"mean return: {np.mean(returns):.1f} +/- {np.std(returns):.1f}")
    print(f"SUCCESS RATE: {100 * outcomes.get('success', 0) / n:.1f}%")
    for k in ("success", "collision", "out_of_bounds", "timeout"):
        print(f"  {k:14s}: {outcomes.get(k, 0)}")


if __name__ == "__main__":
    main()
