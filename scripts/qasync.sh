#!/bin/bash
set -euo pipefail

SRC="ai_hydra/"
DEV_DEST="/opt/dev/ai_hydra/hydra_venv/lib/python3.11/site-packages/ai_hydra/"
QA_DEST="/opt/qa/ai_hydra/hydra_venv/lib/python3.11/site-packages/ai_hydra/"

mkdir -p "$DEV_DEST" "$QA_DEST"

rsync -avr --delete "$SRC" "$DEV_DEST"
rsync -avr --delete "$SRC" "$QA_DEST"
