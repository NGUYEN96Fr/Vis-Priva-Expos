#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition gpu-test
#SBATCH -w node21

#SBATCH --output=/home/users/vnguyen/intern20/Vis-Priva-Expos/privacy/tools/privacy.out
#SBATCH --error=/home/users/vnguyen/intern20/Vis-Priva-Expos/privacy/tools/privacy.errs

python user_selector_ft0.py --config_file ../configs/rf.yaml --model_name best_accom_ft01.pkl --situation ACCOM

python ftop_ft1.py --pre_model out/best_accom_ft0.pkl --model_name best_accom_ft1.pkl