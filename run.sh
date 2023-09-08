#!/bin/bash
#PBS -N hitl_bo
#PBS -j oe
#PBS -o logs.out
#PBS -e logs.out
#PBS -lselect=1:ncpus=64:mem=64gb
#PBS -lwalltime=8:00:00
#PBS -J 1-300

module load anaconda3/personal
cd $PBS_O_WORKDIR
source activate hitl-bo
python3 bo/run.py $PBS_ARRAY_INDEX UCB
python3 bo/run.py $PBS_ARRAY_INDEX EI
python3 bo/plot_regret.py