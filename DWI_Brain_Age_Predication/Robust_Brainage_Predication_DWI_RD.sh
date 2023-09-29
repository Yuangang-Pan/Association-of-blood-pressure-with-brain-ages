#!/bin/bash
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=320GB
#PBS -l wd
#PBS -l walltime=45:00:00
#PBS -P ey6
#PBS -lstorage=scratch/ey6+gdata/ey6

module load cuda/10.1 
module load cudnn/7.6.5-cuda10.1
module load python3/3.7.4

# export PYTHONPATH="/g/data/ey6/Kai/local/lib64/python3.6/site-packages:/g/data/ey6/Kai/local/lib/python3.6/site-packages":$PYTHONPATH

export PYTHONPATH="/home/587/yp7211/.local/lib/python3.7/site-packages:/home/587/yp7211/.local/lib/python3.7/site-packages":$PYTHONPATH

python3 Robust_Brainage_Predication_DWI_RD.py

# module load python3/3.7.4
# pip3 install -U --user scikit-learn

