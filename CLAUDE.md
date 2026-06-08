# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A MuJoCo simulation of the F8C Lightning spacecraft (from Star Citizen) flying through an asteroid belt. A for-fun side project, not a formal submission. See the Roadmap in `README.md` for intended direction (asteroid belt, learning-based controllers, realistic 17-thruster dynamics).

## Setup & Run

Everything (RL + manual play) uses a single **`asteroid-belt-runner`** conda env (Python 3.11:
MuJoCo 3.3.7, Gymnasium 0.29.1, Stable-Baselines3 2.3.0, PyTorch 2.2.1+cu121, pynput, imageio,
**numpy 1.26 / <2** — the RL stack requires it; `main.py` manual play runs fine under it too).
The README quick-start's `requirements.txt` (numpy 2.x) is the **stale** older recipe — don't
follow it; build the env per the RL deps above.

> **Crash note (important):** this conda's torch 2.2.1 intermittently **segfaults while importing
> `torch._dynamo`** (a C-level flake; in 3.10 it surfaced as an `sre_compile` ValueError instead).
> It only bites at import. The fix is **NOT** a Python version (3.11 still flakes) — it's in
> `train/train_ppo.py`: SubprocVecEnv `start_method="forkserver"` + torch/SB3 imports moved into
> `main()` + `_warm_import` retry, with `Agent_tool/train_resilient.sh` auto-resuming the rare
> startup flake. **Do not switch to spawn (constant crashes) or fork (CUDA deadlock).**

Run everything **from the repo root**:

```bash
conda run -n asteroid-belt-runner python main.py                 # manual-play viewer (for fun)
conda run -n asteroid-belt-runner python Agent_tool/check_env.py # env sanity check (headless)
conda run -n asteroid-belt-runner python train/train_ppo.py --timesteps 1_000_000
conda run -n asteroid-belt-runner python Agent_tool/preview_belt.py --n 80   # needs a display
conda run -n asteroid-belt-runner python Agent_tool/rollout_viewer.py --model logs/<run>/best/best_model.zip
tensorboard --logdir logs/
```

No formal test suite/linter. `Agent_tool/check_env.py` is the closest thing to a test — run it
after touching `envs/`.

## Working conventions (IMPORTANT)

This repo follows the `Human_exo_interaction` project's layout for AI-assisted work:
- **`Agent_log/CLAUDE_LOG.md`** — append a dated entry after every substantive action (create/modify/
  **delete**/train/diagnose). Git-tracked & pushed. Rules in `Agent_log/README.md`. Always log when you
  edit this `CLAUDE.md`.
- **`TODO_list/PROJECT_PLAN.md`** — the living 6-phase roadmap; tick boxes as phases complete.
- **`Agent_tool/`** — throwaway/helper scripts (smoke tests, viewers, sanity checks). Git-tracked.
- **`logs/`** — training output / checkpoints / tensorboard. Git-ignored.

## Architecture

The simulation is a single real-time loop in `main.py` driving a MuJoCo model, with input and control logic split into separate modules:

- **`environment.xml`** — the MuJoCo model. One free-jointed `spacecraft` body using the `F8_lightning.stl` mesh, with hand-tuned `<inertial>` (mass 153775 kg, inertia from `STL_inertia_calculator.py`). Gravity is **zero** (space), `timestep="0.004"`. Six `<general>` actuators (`Fx Fy Fz Mx My Mz`) apply force/torque through a single `virtual_thruster` site — this is the *simplified* dynamics model; realistic per-thruster dynamics are future work.

- **`main.py`** — model load, viewer launch, and the step loop. Each iteration: updates the chase camera, polls `manual_controller` flags (clear/reset/quit), gets a velocity command, applies it, steps physics, and periodically calls `utility.report_motion_status`. `USE_MANUAL_CONTROL` toggles keyboard control.

- **`manual_controller.py`** — keyboard/mouse input via `pynput`, running on a background listener thread. All shared state (`velocity_cmd`, `flags`, etc.) is guarded by `ctrl_lock`. The main loop reads it through `control_update_speed` and `check_flag`.

- **`attitude_controller.py`** — `PIController` / `PIDController` classes for three-axis angular-velocity regulation. Currently standalone (not wired into `main.py`'s active path).

- **`utility.py`** — telemetry printing (sim time, linear/angular velocity & acceleration, optionally per-actuator forces).

### Important: two control schemes, and which one is live

The code carries **two** input/control approaches, and large blocks of the older one are commented out rather than deleted. When editing, trace what `main.py` actually calls:

1. **Force/torque control (older, mostly commented out)** — keys map to actuator `ctrl` deltas (`KEY_TO_CTRL_INDEX`), accumulated and clamped to `actuator_ctrlrange`. Driven by `manual_controller.setup` + `control_update`, applied via `_apply_control` / `_apply_angular_velocity`. These paths are **disabled** in the current `main.py`.

2. **Direct velocity control (current, live path)** — `main.py` calls `speed_control_setup` + `control_update_speed`, then `_apply_velocity` writes linear & angular velocity **directly into `data.qvel`**, bypassing the physics actuators. Linear velocity is rotated into the world frame (`R @ vel_cmd`); angular velocity stays in the body frame. Keys: `WASD`+`C/V` translate (`POSITION_KEYS`), `U/O J/L I/K` rotate (`ANGULAR_KEYS`). Space = clear, Backspace = reset, Esc = quit.

Because the live path sets `qvel` directly, the six XML actuators are not currently exercised; they matter for the force-based scheme and future realistic dynamics.

### RL stack (added 2026-06-08)

- **`envs/asteroid_mesh.py`** — procedurally builds the irregular "potato" asteroid mesh library
  (`assets/asteroids/asteroid_*.obj`): icosphere + random Gaussian bumps/craters, pure NumPy. Run
  once to (re)build. MuJoCo collides on the **convex hull** of the mesh (craters filled, bumps kept).
- **`envs/belt_generator.py`** — builds the scene at runtime via `mujoco.MjSpec`: loads
  `environment.xml`, attaches an invisible capsule **collision proxy** to the ship (the STL geom
  stays visual-only), and scatters `BeltConfig.n_asteroids` **free-joint potato-mesh** asteroids in
  an X-axis slab — each a random library mesh with a unique per-axis `scale` (power-law size x aspect)
  and random orientation, min-separation so they don't overlap. Collision masks make asteroids collide
  only with the ship (`SHIP_*`/`AST_*`), not each other or the axis markers. Returns `list[Asteroid]`
  (body/geom/joint names + `r_eff`, a **conservative enclosing-sphere radius**). The scene is compiled
  **once**; the env re-places rocks per episode rather than recompiling.
- **`envs/asteroid_belt_env.py`** — `AsteroidBeltEnv(gym.Env)`. Action = normalized 6-vector mapped
  to **F8C-calibrated** force/torque (sign-asymmetric main/retro thrust + RCS torques; see
  [[f8c-performance-specs]]), distinct from `main.py`'s kinematic play mode; realistic mode -> 17
  thrusters. Obs (160) = body-frame vel/ang-vel + fwd/up + goal dir/dist (16) **plus a body-frame
  full-sphere radar** (2 channels x n_az x n_el: proximity + closing velocity, no occlusion, nearest
  rock per bin). Reward = goal-potential + heading + proximity + spin + **G-load** (penalize >g_safe
  to keep the pilot alive) + ctrl/time, success/collision/OOB terminals. `reset()` re-scatters rocks
  via qpos/qvel (build-once); curriculum density via `n_active` (extra rocks parked far out of bounds).
  The stored MjSpec is `self.mj_spec` (NOT `self.spec`, which Gymnasium reserves for `EnvSpec`).
- **`train/train_ppo.py`** — SB3 PPO over `SubprocVecEnv`, checkpoints/best-model/tensorboard to `logs/`.
- **`envs/thruster_layout.py`** — the 17 realistic thrusters (Roadmap #3), built; training on them is R10.

### Conventions

- Indexing into `data.qvel`/`data.qacc` is always relative to `joint_adr` (the free joint's DOF address): `[adr:adr+3]` = linear, `[adr+3:adr+6]` = angular.
- Angles in the MuJoCo model are in **radians** (`compiler angle="radian"`); `manual_controller` converts deg/sec inputs with `math.radians`.
