#!/bin/bash
#SBATCH -N 8
#SBATCH -n 15
#SBATCH --partition cpu
#SBATCH -w node04, node05, node06, node08

#SBATCH --output=/home/users/vnguyen/privacy.out
#SBATCH --error=/home/users/vnguyen/privacy.errs

python user_expo_eval.py configs.ini