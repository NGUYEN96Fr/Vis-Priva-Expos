#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition bigmem
#SBATCH -w node09
#SBATCH --exclusive
#SBATCH -J FTCV5_CPUs

#SBATCH --output=/home/users/vnguyen/intern20/Vis-Priva-Expos/privacy/tools/FTCV5_CPUs.out
#SBATCH --error=/home/users/vnguyen/intern20/Vis-Priva-Expos/privacy/tools/FTCV5_CPUs.errs

python ft_official.py --config_file ../configs/rf_ft.yaml --model_name bestCV5_bank.pkl --situation BANK

python ft_official.py --config_file ../configs/rf_ft.yaml --model_name bestCV5_accom.pkl --situation ACCOM

python ft_official.py --config_file ../configs/rf_ft.yaml --model_name bestCV5_it.pkl --situation IT

python ft_official.py --config_file ../configs/rf_ft.yaml --model_name bestCV5_wait.pkl --situation WAIT


### Cross validation 5