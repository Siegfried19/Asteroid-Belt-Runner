# Saved models

Curated model snapshots, kept in git (small) so they survive across machines.
(General training output stays in the git-ignored `logs/`.)

## `ppo_traverse_n40_63pct.zip`  ← current best
- PPO controller for the **current "final" architecture** (full-sphere radar obs, F8C-calibrated
  thrust, traverse curriculum), trained on the enlarged belt with the redesigned reward.
- Task: **sparse belt (40 rocks), straight on-axis traverse** (`--exit-r 0 0`), full 700 m distance.
- **Eval (final `model.zip`, 100 episodes, n40, exit-r 0): SUCCESS 63% / out-of-bounds 0% /
  timeout 0% / collision 37%.** Highest the current architecture has reached; the traverse
  curriculum killed the old "fear-of-retreat" out-of-bounds failures. Only remaining failure = collision.
- ⚠️ This is the *easy* task. Random off-axis exits and the dense (135-rock) belt are **not solved**
  (see the v15 note below). Always eval the **final `model.zip`**, not just `best/` — `n_eval_episodes`
  noise once made `best_model` selection pure luck (`Agent_log/REWARD_EXPERIMENTS.md`, 2026-06-09).

Watch it fly (needs a display):
```bash
conda run -n asteroid-belt-runner python Agent_tool/rollout_viewer.py \
    --model models/ppo_traverse_n40_63pct.zip --n-asteroids 40 --exit-r 0 0 --episodes 6
```

## `ppo_v15_best_41pct.zip`
- PPO controller for the **simplified dynamics** asteroid belt (run `ppo_rebuild_v15`, the
  1.5 M-step checkpoint — the policy peaked there before later-training divergence).
- Belt: ~31 big potato-mesh rocks, wide gaps (min_gap 55 → ~58 m, ~2x the ship), oriented-box
  ship collision (wings count), 540 m traverse, no skirt corridor.
- **Eval (100 episodes, full density): SUCCESS 41% / out-of-bounds 0% / collision 59%.**
  Genuinely threads the gaps (0% skirting); failures are clipping a rock mid-traverse.

> ⚠️ **Status note (2026-06-09):** this 41% is from the **earlier, simpler task** (straight-line
> traverse, *no off-axis exit*; note oob 0%) — superseded as the headline model by
> `ppo_traverse_n40_63pct.zip` above. Re-evaluated under the *current* env this old v15 model scores
> only **~2%** (it never saw the radar obs / F8C thrust). The **straight-traverse** task is now solved
> at 63% (see above); the **hard task still open** = random off-axis exits + dense (135-rock) belt,
> where PPO collapses into a "ram straight through" local optimum (the dense progress reward outweighs
> avoidance). Full diagnosis + attack plan: `Agent_log/REWARD_EXPERIMENTS.md` (2026-06-09).

Watch it fly (needs a display):
```bash
Agent_tool/watch.sh models/ppo_v15_best_41pct.zip 30 40
```
Re-evaluate headless:
```bash
conda run -n asteroid-belt-runner python Agent_tool/eval_policy.py \
    --model models/ppo_v15_best_41pct.zip --episodes 100 --n-asteroids 40
```
Next steps to push past 41%: early-stopping / lower LR (the run diverged after ~1.5 M), and
tuning the proximity reward to cut the collision rate. See `Agent_log/CLAUDE_LOG.md`.
