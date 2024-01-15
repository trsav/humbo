#!/bin/bash

# Define source and destination
SRC="trs20@login.hpc.ic.ac.uk:llmbo/bo/benchmark_results_specific/"
DEST="bo/benchmark_results_specific/"

# Use rsync to copy only res.json files and their containing folders
rsync -avm \
  --include='*/' \
  --include='res.json' \
  --exclude='*' \
  "$SRC" "$DEST"
