#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition gpu-test
#SBATCH -w node21

#SBATCH --output=/home/users/vnguyen/privacy.out
#SBATCH --error=/home/users/vnguyen/privacy.errs

python main.py configs.ini