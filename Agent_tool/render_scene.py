"""Headless render of the rebuilt asteroid-belt scene to PNGs (no display needed).

Run from repo root:
    MUJOCO_GL=egl conda run -n asteroid-belt-runner python Agent_tool/render_scene.py --seed 3

Writes images/scene_*.png — an overview of the belt, a ship close-up (to verify
heading), and a down-axis "pilot" view.
"""
import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import imageio.v2 as imageio
import mujoco
import numpy as np

from envs.asteroid_belt_env import AsteroidBeltEnv

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")


def shot(renderer, data, lookat, dist, az, el, path):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance = dist
    cam.azimuth = az
    cam.elevation = el
    renderer.update_scene(data, camera=cam)
    img = renderer.render()
    imageio.imwrite(path, img)
    print(f"  wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=3)
    args = p.parse_args()

    env = AsteroidBeltEnv()
    env.reset(seed=args.seed)
    m, d = env.model, env.data

    x0, x1 = env.base_cfg.belt_x_range
    cx = 0.5 * (x0 + x1)
    print(f"belt_x_range=({x0},{x1})  belt center x={cx}  goal_x={env.goal_x}  ngeom={m.ngeom}")

    m.vis.global_.offwidth = 1600
    m.vis.global_.offheight = 900
    # hide the translucent green goal/spawn marker so it doesn't occlude the ship
    gm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "goal_marker")
    if gm >= 0:
        m.geom_rgba[gm, 3] = 0.0
    # the belt's large extent pushes the near clip plane out ~tens of metres, which
    # clips close-up ship shots to black; shrink it so dist~38 m views render.
    m.vis.map.znear = 0.002
    r = mujoco.Renderer(m, height=900, width=1600)
    os.makedirs(OUT, exist_ok=True)

    span = x1 - x0
    # 1. hero: oblique bird's-eye of the whole belt (ship at origin + belt depth)
    shot(r, d, [cx, 0, 0], dist=1.9 * span, az=55, el=-24,
         path=os.path.join(OUT, "scene_hero.png"))
    # 2. chase: behind/above the ship (-X) looking forward into the belt (+X)
    shot(r, d, [0.35 * x1, 0, 0], dist=1.0 * span, az=0, el=-12,
         path=os.path.join(OUT, "scene_chase.png"))
    # 3. down-axis pilot view: looking straight along the belt axis
    shot(r, d, [cx, 0, 0], dist=1.1 * span, az=180, el=-4,
         path=os.path.join(OUT, "scene_downaxis.png"))

    r.close()
    print("done")


if __name__ == "__main__":
    main()
