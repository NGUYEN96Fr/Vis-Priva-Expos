#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition bigmem
#SBATCH -w node09
#SBATCH --exclusive
#SBATCH -J MOBINET_CV5_CPUs

#SBATCH --output=/home/users/vnguyen/intern20/Vis-Priva-Expos/privacy/tools/MOBINET_CV5_CPUs.out
#SBATCH --error=/home/users/vnguyen/intern20/Vis-Priva-Expos/privacy/tools/MOBINET_CV5_CPUs.errs

python ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name bestCV5_mobinet_bank.pkl --situation BANK

python ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name bestCV5_mobinet_accom.pkl --situation ACCOM

python ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name bestCV5_mobinet_it.pkl --situation IT

python ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name bestCV5_mobinet_wait.pkl --situation WAIT


### Cross validation 5