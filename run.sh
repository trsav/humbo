#!/bin/bash
#PBS -N hitl_bo
#PBS -j oe
#PBS -o logs.out
#PBS -e logs.out
#PBS -lselect=1:ncpus=48:mem=110gb
#PBS -lwalltime=8:00:00

module load anaconda3/personal

cd $PBS_O_WORKDIR
source activate hitl-bo

python3 bo/main.py UCB 1 
# mpiexec -n 48 python3 bo/run.py EI 1 
# mpiexec -n 48 python3 bo/run.py UCB 2 
# mpiexec -n 48 python3 bo/run.py UCB 5 

# python3 bo/plot_regret.py