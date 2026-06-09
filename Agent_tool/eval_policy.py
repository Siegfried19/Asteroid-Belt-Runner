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
    p.add_argument("--max-steps", type=int, default=None,
                   help="episode step budget (default: generous, auto-scaled with --belt-len)")
    p.add_argument("--belt-len", type=float, default=None, help="far edge of the belt (m); match training")
    p.add_argument("--goal-mode", choices=["traverse", "interior_point"], default="traverse",
                   help="match the mode the model was trained on")
    p.add_argument("--arrival-speed", type=float, default=None,
                   help="interior_point: require speed <= this on arrival (match training tier)")
    p.add_argument("--exit-r", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                   help="override exit off-axis range for eval (default: env default)")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=5000)
    args = p.parse_args()

    belt_far = args.belt_len if args.belt_len else BeltConfig().belt_x_range[1]
    max_steps = args.max_steps if args.max_steps else max(2200, int(round(2200 * belt_far / 700.0)))
    cfg = BeltConfig(n_asteroids=args.n_asteroids, belt_x_range=(100.0, belt_far), seed=args.seed)
    env = AsteroidBeltEnv(cfg=cfg, dynamics=args.dynamics, max_steps=max_steps,
                          goal_mode=args.goal_mode, arrival_speed=args.arrival_speed,
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
