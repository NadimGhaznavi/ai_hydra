#!/bin/bash
#

SRC=ai_hydra
DEV_DEST=/opt/dev/ai_hydra/hydra_venv/lib/python3.11/site-packages/ai_hydra
QA_DEST=/opt/qa/ai_hydra/hydra_venv/lib/python3.11/site-packages/ai_hydra


rm -rf $DEV_DEST
mkdir $DEV_DEST
rsync -avr --delete $SRC/* $DEV_DEST


rm -rf $QA_DEST
mkdir $QA_DEST
rsync -avr --delete $SRC/* $QA_DEST
