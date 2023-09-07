#!/bin/bash
#PBS -N hitl_bo
#PBS -j oe
#PBS -o output_logs.out
#PBS -e error_logs.out
#PBS -lselect=1:ncpus=16:mem=64gb
#PBS -lwalltime=4:00:00
#PBS -J 1-80

module load anaconda3/personal
cd $PBS_O_WORKDIR
source activate hitl-bo
python3 bo/problem_setup.py
python3 bo/run.py $PBS_ARRAY_INDEX