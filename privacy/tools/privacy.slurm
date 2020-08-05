#!/bin/bash
#SBATCH -N 3
#SBATCH -n 20
#SBATCH --partition cpu
#SBATCH -w node03, node07, node08

#SBATCH --output=/home/users/vnguyen/privacy.out
#SBATCH --error=/home/users/vnguyen/privacy.errs

python main.py configs.ini