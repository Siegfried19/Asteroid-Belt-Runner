# Saved models

Curated model snapshots, kept in git (small) so they survive across machines.
(General training output stays in the git-ignored `logs/`.)

## `ppo_v15_best_41pct.zip`
- PPO controller for the **simplified dynamics** asteroid belt (run `ppo_rebuild_v15`, the
  1.5 M-step checkpoint — the policy peaked there before later-training divergence).
- Belt: ~31 big potato-mesh rocks, wide gaps (min_gap 55 → ~58 m, ~2x the ship), oriented-box
  ship collision (wings count), 540 m traverse, no skirt corridor.
- **Eval (100 episodes, full density): SUCCESS 41% / out-of-bounds 0% / collision 59%.**
  Genuinely threads the gaps (0% skirting); failures are clipping a rock mid-traverse.

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
