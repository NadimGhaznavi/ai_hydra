#!/bin/bash
set -euo pipefail

SRC="ai_hydra/"
DEV_DEST="/opt/dev/ai_hydra/hydra_venv/lib/python3.11/site-packages/ai_hydra/"

mkdir -p "$DEV_DEST"

rsync -avr --delete "$SRC" "$DEV_DEST"
