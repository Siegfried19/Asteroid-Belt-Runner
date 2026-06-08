#!/usr/bin/env bash
# Launch the live MuJoCo viewer to watch a trained policy fly the belt.
# Wraps the DISPLAY/XAUTHORITY + conda-run boilerplate so it's one command.
#
# Usage (from repo root):
#   Agent_tool/watch.sh                                  # default: v4 best model, 30 episodes
#   Agent_tool/watch.sh logs/<run>/best/best_model.zip   # a specific model
#   Agent_tool/watch.sh logs/<run>/best/best_model.zip 10 100   # model, episodes, n_asteroids
#
# In the viewer: drag = rotate, scroll = zoom, close the window to stop.
set -e
cd "$(dirname "$0")/.."

MODEL="${1:-logs/ppo_rebuild_v4/best/best_model.zip}"
EPISODES="${2:-30}"
N_AST="${3:-60}"

export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"

exec conda run -n asteroid-belt-runner python Agent_tool/rollout_viewer.py \
  --model "$MODEL" --episodes "$EPISODES" --n-asteroids "$N_AST"
