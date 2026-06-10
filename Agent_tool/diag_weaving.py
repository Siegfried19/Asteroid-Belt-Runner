"""Does the 63% straight-traverse model actually WEAVE, or does it just fly straight and
win whenever the X-axis happens to be clear? Two tests:

  Part A (clear-lane correlation): N normal n40/exit0 episodes. Per episode record the
    outcome, the min clearance of any rock to the straight start->goal axis line, and the
    max lateral deviation (rho_yz) the ship ever reached. If the model only flies straight,
    successes will cluster where the axis was clear, and max lateral deviation stays tiny.

  Part B (forced block, causal): park all rocks but ONE, placed dead on the axis at x=x_block.
    The ship MUST weave around a single on-axis rock to reach the goal. If it can weave, it
    passes; if it only flies straight, it collides. Run a few x_block positions.

Run from repo root:
    conda run -n asteroid-belt-runner python Agent_tool/diag_weaving.py \
        --model models/ppo_traverse_n40_63pct.zip
"""
import argparse
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import numpy as np
from stable_baselines3 import PPO

from envs.asteroid_belt_env import AsteroidBeltEnv
from envs.belt_generator import BeltConfig


def axis_clearance(env):
    """Min surface clearance of any active rock to the straight line from the ship start
    (origin) to the goal. With exit_r=0 the goal is on the X axis, so this is essentially
    min(rho_yz - r_eff) over rocks whose x lies between start and goal."""
    centers, r = env._active_centers()
    if len(centers) == 0:
        return np.inf
    g = env.goal
    a = np.zeros(3)              # ship start
    ab = g - a
    t = np.clip((centers - a) @ ab / (ab @ ab), 0.0, 1.0)   # projection param onto segment
    proj = a + t[:, None] * ab
    d = np.linalg.norm(centers - proj, axis=1) - r          # surface distance to the path
    return float(d.min())


def run_episode(env, model, seed, max_steps):
    obs, _ = env.reset(seed=seed)
    clear = axis_clearance(env)
    max_lat, done, ret = 0.0, False, 0.0
    steps = 0
    while not done and steps < max_steps:
        a, _ = model.predict(obs, deterministic=True)
        obs, rwd, term, trunc, info = env.step(a)
        max_lat = max(max_lat, float(np.linalg.norm(env.ship_pos[1:3])))
        ret += rwd
        done = term or trunc
        steps += 1
    return info.get("outcome", "?"), clear, max_lat


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/ppo_traverse_n40_63pct.zip")
    p.add_argument("--n-asteroids", type=int, default=40)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=2200)
    p.add_argument("--seed", type=int, default=5000)
    args = p.parse_args()

    cfg = BeltConfig(n_asteroids=args.n_asteroids, seed=args.seed)
    env = AsteroidBeltEnv(cfg=cfg, max_steps=args.max_steps, randomize_belt=True)
    env.set_exit_r(0.0, 0.0)                     # straight exit = how this model was trained
    model = PPO.load(args.model, device="cpu")
    block_thresh = float(env.ship_box_half[1]) + 5.0   # axis "blocked" if a rock is within this of the path
    print(f"model: {args.model}  n={args.n_asteroids}  block_thresh(axis)={block_thresh:.1f} m\n")

    # ---- Part A: clear-lane correlation -------------------------------------------------
    rows = []
    for ep in range(args.episodes):
        rows.append(run_episode(env, model, args.seed + ep, args.max_steps))
    outc = [r[0] for r in rows]
    clear = np.array([r[1] for r in rows])
    lat = np.array([r[2] for r in rows])
    succ = np.array([o == "success" for o in outc])
    n = len(rows)

    print("=== Part A: clear-lane correlation (n40, exit0) ===")
    print(f"overall success: {100*succ.mean():.0f}%  "
          f"({sum(o=='success' for o in outc)} succ / {sum(o=='collision' for o in outc)} coll / "
          f"{sum(o=='out_of_bounds' for o in outc)} oob / {sum(o=='timeout' for o in outc)} to)")
    axis_clear = clear > block_thresh            # straight tube was passable
    for label, mask in [("axis CLEAR (straight passable)", axis_clear),
                        ("axis BLOCKED (rock on path)   ", ~axis_clear)]:
        if mask.sum():
            print(f"  {label}: {mask.sum():3d} eps  success {100*succ[mask].mean():4.0f}%  "
                  f"mean max-lateral {lat[mask].mean():5.1f} m")
    if succ.any():
        print(f"  successes' max-lateral: mean {lat[succ].mean():.1f} m, median {np.median(lat[succ]):.1f} m, "
              f"max {lat[succ].max():.1f} m   (large => it weaves; small => it flies straight)")
    else:
        print("  successes' max-lateral: n/a (no successes in Part A)")
    print(f"  belt rim is {cfg.belt_yz_radius:.0f} m; ship can deviate up to ~"
          f"{cfg.belt_yz_radius + env.oob_yz_margin:.0f} m before OOB\n")

    # ---- Part B: forced single on-axis rock (causal) ------------------------------------
    print("=== Part B: forced block — ONE rock dead on the axis, must weave around it ===")
    for x_block in (200.0, 350.0, 500.0):
        outs = []
        for k in range(20):
            obs, _ = env.reset(seed=10_000 + k)
            env.n_active = 1                                   # only rock 0 is live
            a0 = env.ast_qpos_adr[0]
            env.data.qpos[a0:a0 + 3] = [x_block, 0.0, 0.0]     # plant it on the straight path
            env.data.qvel[env.ast_dof_adr[0]:env.ast_dof_adr[0] + 6] = 0.0
            mujoco.mj_forward(env.model, env.data)
            obs = env._obs()
            done, steps = False, 0
            while not done and steps < args.max_steps:
                act, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(act)
                done = term or trunc
                steps += 1
            outs.append(info.get("outcome", "?"))
        s = sum(o == "success" for o in outs)
        c = sum(o == "collision" for o in outs)
        print(f"  rock@x={x_block:.0f} (r_eff~{env.ast_r[0]:.0f} m): "
              f"success {100*s/len(outs):3.0f}%  ({s} succ / {c} coll / "
              f"{sum(o=='out_of_bounds' for o in outs)} oob / {sum(o=='timeout' for o in outs)} to)")
    env.close()


if __name__ == "__main__":
    main()
