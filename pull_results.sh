#!/bin/bash

# first try to remove bo/benchmark_results_rkhs
rm -rf bo/benchmark_results_rkhs

# Define source and destination
SRC="trs20@login.hpc.ic.ac.uk:llmbo/bo/benchmark_results_rkhs/"
DEST="bo/benchmark_results_rkhs/"

# Use rsync to copy only res.json files and their containing folders
rsync -avm \
  --include='*/' \
  --include='res.json' \
  --exclude='*' \
  "$SRC" "$DEST"
