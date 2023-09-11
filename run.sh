#!/bin/bash
#PBS -N hitl_bo
#PBS -j oe
#PBS -o logs.out
#PBS -e logs.out
#PBS -lselect=1:ncpus=6:mem=64gb
#PBS -lwalltime=12:00:00
#PBS -J 1-50

module load anaconda3/personal
cd $PBS_O_WORKDIR
source activate hitl-bo

# foe each aq function, run 50 trials, each function evaluated for each behaviour
python3 bo/run.py $PBS_ARRAY_INDEX UCB 1
python3 bo/run.py $PBS_ARRAY_INDEX EI 1
python3 bo/run.py $PBS_ARRAY_INDEX UCB 2
python3 bo/run.py $PBS_ARRAY_INDEX EI 2
python3 bo/run.py $PBS_ARRAY_INDEX UCB 5
python3 bo/run.py $PBS_ARRAY_INDEX EI 5

# python3 bo/plot_regret.py