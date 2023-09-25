#!/bin/bash
#PBS -N hitl_bo
#PBS -j oe
#PBS -o logs.out
#PBS -e logs.out
#PBS -lselect=1:ncpus=6:mem=16gb
#PBS -lwalltime=16:00:00
#PBS -J 1-128

module load anaconda3/personal

cd $PBS_O_WORKDIR
source activate multi_fidelity_experimental_design_env

python3 bo/benchmark_human_specific.py $PBS_ARRAY_INDEX