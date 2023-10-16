#!/bin/bash

# Define source and destination
SRC="trs20@login.hpc.ic.ac.uk:HITL-BO/bo/benchmark_results_rkhs/"
DEST="bo/benchmark_results_rkhs/"

# Use rsync to copy only res.json files and their containing folders
rsync -avm \
  --include='*/' \
  --include='res.json' \
  --exclude='*' \
  "$SRC" "$DEST"
