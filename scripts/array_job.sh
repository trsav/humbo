#!/bin/bash
#PBS -j oe
#PBS -lselect=1:ncpus=1:mem=16gb
#PBS -lwalltime=24:00:00

module load anaconda3/personal
cd $PBS_O_WORKDIR
source activate multi_fidelity_experimental_design_env

python3 bo/benchmarks.py --function $function --array_index $PBS_ARRAY_INDEX --behaviour_index $B
