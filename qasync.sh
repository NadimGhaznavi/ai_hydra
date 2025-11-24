#!/bin/bash 
#

SRC=ai_hydra
DEST=/opt/dev/ai_hydra/hydra_venv/lib/python3.11/site-packages/ai_hydra

rm -rf $DEST
mkdir $DEST
rsync -avr --delete $SRC/* $DEST
