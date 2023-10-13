#!/bin/bash
#PBS -N $UUID
#PBS -j oe
#PBS -o logs_$UUID.out
#PBS -e logs_$UUID.out
#PBS -lselect=1:ncpus=1:mem=16gb
#PBS -lwalltime=24:00:00
#PBS -J 1-$A_TOTAL

module load anaconda3/personal
cd $PBS_O_WORKDIR
source activate multi_fidelity_experimental_design_env

python3 bo/benchmark.py --function $function $PBS_ARRAY_INDEX $B