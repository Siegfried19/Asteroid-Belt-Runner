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
from envs.belt_generator import BeltConfig, warm_belt_cache


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
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

    class CurriculumCallback(BaseCallback):
        """Ramp difficulty over the first `ramp_frac` of training, easy -> hard:
        belt density n_start -> n_end, AND exit off-axis distance er_start -> er_end (each a
        (min, max) pair). The agent first learns to fly straight through a sparse belt, then to
        weave through a dense one to a far off-axis exit. Updates fire when the (discrete) density
        steps; exit_r rides the same steps (fine-grained enough since n spans ~100 levels)."""

        def __init__(self, total_timesteps, n_start, n_end, er_start, er_end,
                     tx_start, tx_end, ramp_frac=0.6, verbose=0):
            super().__init__(verbose)
            self.total = total_timesteps
            self.n_start, self.n_end = n_start, n_end
            self.er_start, self.er_end = er_start, er_end
            self.tx_start, self.tx_end = tx_start, tx_end
            self.ramp_frac = ramp_frac
            self._cur = None

        def _on_step(self):
            frac = min(1.0, self.num_timesteps / (self.total * self.ramp_frac))
            n = int(round(self.n_start + frac * (self.n_end - self.n_start)))
            er_min = self.er_start[0] + frac * (self.er_end[0] - self.er_start[0])
            er_max = self.er_start[1] + frac * (self.er_end[1] - self.er_start[1])
            tx = self.tx_start + frac * (self.tx_end - self.tx_start)
            if n != self._cur:
                self.training_env.env_method("set_n_asteroids", n)
                self.training_env.env_method("set_exit_r", er_min, er_max)
                self.training_env.env_method("set_traverse", tx)
                self._cur = n
                if self.verbose:
                    print(f"[curriculum] n={n} exit_r=({er_min:.0f},{er_max:.0f}) traverse={tx:.0f} "
                          f"at {self.num_timesteps} steps")
            return True

    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=20)
    p.add_argument("--n-asteroids", type=int, default=135)
    p.add_argument("--dynamics", choices=["simplified", "realistic"], default="simplified")
    p.add_argument("--curriculum", action="store_true",
                   help="ramp belt density + exit off-axis distance from easy to hard")
    p.add_argument("--n-start", type=int, default=30, help="curriculum starting density")
    p.add_argument("--exit-r-end", type=float, nargs=2, default=(90.0, 150.0),
                   metavar=("MIN", "MAX"), help="curriculum target exit off-axis range (match env)")
    p.add_argument("--ramp-frac", type=float, default=0.6,
                   help="fraction of training over which curriculum ramps easy->hard")
    p.add_argument("--ent-coef", type=float, default=0.0,
                   help="PPO entropy bonus (raise to encourage exploration)")
    p.add_argument("--net-width", type=int, default=256, help="MLP hidden width (both layers)")
    p.add_argument("--traverse", type=float, nargs=2, default=None, metavar=("START", "END"),
                   help="curriculum ramp of goal distance, e.g. 300 700 (default: no ramp, full traverse)")
    p.add_argument("--vecnorm", action="store_true", help="wrap envs in VecNormalize (reward-scale norm)")
    p.add_argument("--max-steps", type=int, default=2200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", type=str, default="ppo_simplified")
    p.add_argument("--logdir", type=str, default="logs")
    p.add_argument("--checkpoint-freq", type=int, default=50_000)
    p.add_argument("--resume", action="store_true",
                   help="continue from the latest checkpoint in the run dir (for crash recovery)")
    args = p.parse_args()

    run_dir = os.path.join(args.logdir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Warm the belt-layout cache in THIS (single) process before spawning SubprocVecEnv
    # workers. At high density the per-worker rejection sampling tripped a numpy memory-
    # corruption flake when 16 workers ran it concurrently; precomputing once means workers
    # just load the layout. Warm both the train (seed) and eval (seed+1000) belts.
    print(f"[train] warming belt cache (n={args.n_asteroids}) ...", flush=True)
    warm_belt_cache(BeltConfig(n_asteroids=args.n_asteroids, seed=args.seed))
    warm_belt_cache(BeltConfig(n_asteroids=args.n_asteroids, seed=args.seed + 1000))

    env = make_vec_env(
        make_env_fn(args.n_asteroids, args.max_steps, args.seed, args.dynamics),
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
        # NOTE: deliberately NOT passing start_method -> use SB3's safe default ("forkserver" on
        # Linux): a clean server process imports the env stack ONCE and forks workers from it.
        # Earlier this was overridden to "spawn", which re-imports torch in all 16 workers and
        # constantly tripped torch 2.2.1's flaky `_dynamo` import -> the whole crash saga. Do NOT
        # override this (spawn = re-import storm; fork = CUDA-init deadlock). Keeping torch/SB3
        # imports inside main() also keeps the forked workers torch/CUDA-free.
    )
    env = VecMonitor(env, filename=os.path.join(run_dir, "monitor"))
    if args.vecnorm:
        env = VecNormalize(env, norm_obs=False, norm_reward=True, gamma=0.995)  # reward-scale norm

    eval_env = make_vec_env(
        make_env_fn(args.n_asteroids, args.max_steps, args.seed + 1000, args.dynamics),
        n_envs=1,
        seed=args.seed + 1000,
    )
    eval_env = VecMonitor(eval_env)
    if args.vecnorm:
        # must mirror the training env's wrapping; EvalCallback syncs the running stats into it.
        eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False, gamma=0.995)

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
            ent_coef=args.ent_coef,
            learning_rate=3e-4,
            clip_range=0.2,
            n_epochs=10,
            policy_kwargs=dict(net_arch=[args.net_width, args.net_width]),
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
            # 10 eps had std ~+/-800 -> best_model selection was pure noise (once saved a lucky 50k
            # checkpoint as "best" while the genuinely-good final model scored worse on its 10 eps).
            # 40 eps tightens the SEM enough to pick the real best. Also: the final model.zip is
            # often as good or better -- always eval BOTH best/ and the final model.zip.
            n_eval_episodes=40,
            deterministic=True,
        ),
    ]
    if args.curriculum:
        tx_far = BeltConfig().belt_x_range[1]
        tx_s, tx_e = tuple(args.traverse) if args.traverse else (tx_far, tx_far)
        callbacks.append(
            CurriculumCallback(args.timesteps, args.n_start, args.n_asteroids,
                               er_start=(0.0, 0.0), er_end=tuple(args.exit_r_end),
                               tx_start=tx_s, tx_end=tx_e,
                               ramp_frac=args.ramp_frac, verbose=1)
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
