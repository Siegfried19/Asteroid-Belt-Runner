"""Train a PPO controller to fly the F8C through the asteroid belt (simplified dynamics).

Run from repo root:
    conda run -n asteroid-belt-runner python train/train_ppo.py --timesteps 1_000_000

Outputs (git-ignored) go under logs/<run-name>/:
    tensorboard logs, periodic checkpoints, best model, and the final model.zip.
Watch training: tensorboard --logdir logs/
"""
import argparse
import os
import sys

# Pin every process to a single math/threading lane BEFORE importing numpy/torch/mujoco.
# With 16 SubprocVecEnv workers each spawning OpenMP/BLAS threads (+ the main process's
# torch threads) on a many-core CPU, thread oversubscription caused rare cross-thread
# memory corruption (segfaults / "numpy has no attribute ..." during env reset). One
# lane per process removes the races and also improves throughput. Must precede imports.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
# torch 2.2.1 + Python 3.10 has a flaky TorchDynamo regex-compile bug (sre_compile raises
# "too many values to unpack" mid-run). We never use torch.compile, so disable Dynamo outright.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NOTE: torch / Stable-Baselines3 are imported INSIDE main(), not here. SubprocVecEnv "spawn"
# workers re-import this module to get make_env_fn; keeping torch out of module scope means the
# workers never import torch (or its flaky TorchDynamo) at all -- only the env stack (gym/mujoco).
from envs.asteroid_belt_env import AsteroidBeltEnv
from envs.belt_generator import BeltConfig


def make_env_fn(n_asteroids, max_steps, seed, dynamics):
    def _init():
        cfg = BeltConfig(n_asteroids=n_asteroids, seed=seed)
        return AsteroidBeltEnv(cfg=cfg, dynamics=dynamics, max_steps=max_steps, randomize_belt=True)
    return _init


def main():
    # All torch / SB3 imports live here (main process only; workers never run main).
    import importlib
    import torch
    torch.set_num_threads(1)

    # This env's CPython has a flaky sre_compile (regex compile) that intermittently throws
    # "too many values to unpack (expected 0)" while a library compiles regexes AT IMPORT
    # (seen in torch._dynamo's skipfiles and matplotlib's rcParams). It only bites at import,
    # so eagerly import the known offenders here with retry (purging partial modules) -> once
    # cached, the later lazy import during training is a no-op and cannot crash mid-run.
    def _warm_import(name, tries=20):
        for _ in range(tries):
            try:
                return importlib.import_module(name)
            except Exception:
                for _k in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
                    del sys.modules[_k]
        return None

    _warm_import("torch._dynamo")
    _warm_import("matplotlib")

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    class CurriculumCallback(BaseCallback):
        """Linearly ramp belt density from n_start to n_end over the first `ramp_frac` of training."""

        def __init__(self, total_timesteps, n_start, n_end, ramp_frac=0.6, verbose=0):
            super().__init__(verbose)
            self.total = total_timesteps
            self.n_start, self.n_end = n_start, n_end
            self.ramp_frac = ramp_frac
            self._cur = None

        def _on_step(self):
            frac = min(1.0, self.num_timesteps / (self.total * self.ramp_frac))
            n = int(round(self.n_start + frac * (self.n_end - self.n_start)))
            if n != self._cur:
                self.training_env.env_method("set_n_asteroids", n)
                self._cur = n
                if self.verbose:
                    print(f"[curriculum] n_asteroids -> {n} at {self.num_timesteps} steps")
            return True

    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-asteroids", type=int, default=60)
    p.add_argument("--dynamics", choices=["simplified", "realistic"], default="simplified")
    p.add_argument("--curriculum", action="store_true",
                   help="ramp belt density from --n-start up to --n-asteroids")
    p.add_argument("--n-start", type=int, default=15, help="curriculum starting density")
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", type=str, default="ppo_simplified")
    p.add_argument("--logdir", type=str, default="logs")
    p.add_argument("--checkpoint-freq", type=int, default=50_000)
    p.add_argument("--resume", action="store_true",
                   help="continue from the latest checkpoint in the run dir (for crash recovery)")
    args = p.parse_args()

    run_dir = os.path.join(args.logdir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    env = make_vec_env(
        make_env_fn(args.n_asteroids, args.max_steps, args.seed, args.dynamics),
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
        # "forkserver": a clean server process (no torch/CUDA, since those live in main()) imports
        # the env stack ONCE and forks the workers from it. Avoids both (a) "spawn" re-importing in
        # every worker -> hammering this env's flaky sre_compile, and (b) "fork" deadlocking children
        # of the CUDA-initialised main process. Workers stay torch/CUDA-free.
        vec_env_kwargs=dict(start_method="forkserver"),
    )
    env = VecMonitor(env, filename=os.path.join(run_dir, "monitor"))

    eval_env = make_vec_env(
        make_env_fn(args.n_asteroids, args.max_steps, args.seed + 1000, args.dynamics),
        n_envs=1,
        seed=args.seed + 1000,
    )
    eval_env = VecMonitor(eval_env)

    # crash recovery: if --resume and a checkpoint exists, continue from the latest one
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    latest_ckpt, done_steps = None, 0
    if args.resume and os.path.isdir(ckpt_dir):
        import glob, re
        ckpts = glob.glob(os.path.join(ckpt_dir, "ppo_*_steps.zip"))
        if ckpts:
            latest_ckpt = max(ckpts, key=lambda f: int(re.search(r"_(\d+)_steps", f).group(1)))
            done_steps = int(re.search(r"_(\d+)_steps", latest_ckpt).group(1))

    if latest_ckpt:
        print(f"[train] resuming from {latest_ckpt} (~{done_steps} steps done)")
        model = PPO.load(latest_ckpt, env=env, device="auto", tensorboard_log=run_dir)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=2048,
            batch_size=2048,
            gae_lambda=0.95,
            gamma=0.995,
            ent_coef=0.0,
            learning_rate=3e-4,
            clip_range=0.2,
            n_epochs=10,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=1,
            tensorboard_log=run_dir,
            seed=args.seed,
        )

    callbacks = [
        CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.n_envs, 1),
            save_path=os.path.join(run_dir, "checkpoints"),
            name_prefix="ppo",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(run_dir, "best"),
            log_path=run_dir,
            eval_freq=max(50_000 // args.n_envs, 1),
            n_eval_episodes=10,
            deterministic=True,
        ),
    ]
    if args.curriculum:
        callbacks.append(
            CurriculumCallback(args.timesteps, args.n_start, args.n_asteroids, verbose=1)
        )

    remaining = max(args.timesteps - done_steps, 0)
    model.learn(total_timesteps=remaining, callback=callbacks, progress_bar=False,
                reset_num_timesteps=not latest_ckpt)
    model.save(os.path.join(run_dir, "model"))
    print(f"[train] saved final model to {run_dir}/model.zip")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
