#!/bin/bash

# first try to remove bo/benchmark_results_rkhs
rm -rf bo/benchmark_results_specific_alternatives

# Define source and destination
SRC="trs20@login.hpc.ic.ac.uk:humbo/bo/benchmark_results_specific_alternatives/"
DEST="bo/benchmark_results_specific_alternatives/"

# Use rsync to copy only res.json files and their containing folders
rsync -avm \
  --include='*/' \
  --include='res.json' \
  --exclude='*' \
  "$SRC" "$DEST"
