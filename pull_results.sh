#!/bin/bash

# Define source and destination
SRC="trs20@login.hpc.ic.ac.uk:HITL-BO/bo/benchmark_llmbo_results/"
DEST="bo/benchmark_llmbo_results/"

# Use rsync to copy only res.json files and their containing folders
rsync -avm \
  --include='*/' \
  --include='res.json' \
  --exclude='*' \
  "$SRC" "$DEST"
