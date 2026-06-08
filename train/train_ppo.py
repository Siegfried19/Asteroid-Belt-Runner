"""Train a PPO controller to fly the F8C through the asteroid belt (simplified dynamics).

Run from repo root:
    conda run -n space-robotics-project python train/train_ppo.py --timesteps 1_000_000

Outputs (git-ignored) go under logs/<run-name>/:
    tensorboard logs, periodic checkpoints, best model, and the final model.zip.
Watch training: tensorboard --logdir logs/
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from envs.asteroid_belt_env import AsteroidBeltEnv
from envs.belt_generator import BeltConfig


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


def make_env_fn(n_asteroids, max_steps, seed, dynamics):
    def _init():
        cfg = BeltConfig(n_asteroids=n_asteroids, seed=seed)
        return AsteroidBeltEnv(cfg=cfg, dynamics=dynamics, max_steps=max_steps, randomize_belt=True)
    return _init


def main():
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
    args = p.parse_args()

    run_dir = os.path.join(args.logdir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    env = make_vec_env(
        make_env_fn(args.n_asteroids, args.max_steps, args.seed, args.dynamics),
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
    )
    env = VecMonitor(env, filename=os.path.join(run_dir, "monitor"))

    eval_env = make_vec_env(
        make_env_fn(args.n_asteroids, args.max_steps, args.seed + 1000, args.dynamics),
        n_envs=1,
        seed=args.seed + 1000,
    )
    eval_env = VecMonitor(eval_env)

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

    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False)
    model.save(os.path.join(run_dir, "model"))
    print(f"[train] saved final model to {run_dir}/model.zip")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
