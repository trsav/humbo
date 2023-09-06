#!/bin/bash
#PBS -N hitl_bo
#PBS -j oe
#PBS -o logs.out
#PBS -e logs.out
#PBS -lselect=1:ncpus=16:mem=64gb
#PBS -lwalltime=4:00:00
#PBS -J 1-4

module load anaconda3/personal
cd $PBS_O_WORKDIR
source activate hitl-bo
python3 bo/run.py $(sed -n "${PBS_ARRAY_INDEX}p" bo/problem_data/file_names.txt)