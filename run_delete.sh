#!/bin/bash
#PBS -N hitl_bo
#PBS -j oe
#PBS -o logs.out
#PBS -e logs.out
#PBS -lselect=1:ncpus=6:mem=16gb
#PBS -lwalltime=2:00:00

module load anaconda3/personal

cd $PBS_O_WORKDIR
source activate multi_fidelity_experimental_design_env

python3 bo/utils.py