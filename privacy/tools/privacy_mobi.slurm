#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition bigmem
#SBATCH -w node09
#SBATCH --exclusive
#SBATCH -J FT

#SBATCH --output=/home/users/vnguyen/intern20/Vis-Priva-Expos/privacy/tools/ft_mobi.out
#SBATCH --error=/home/users/vnguyen/intern20/Vis-Priva-Expos/privacy/tools/ft_mobi.errs

python ft0_user_selector.py --config_file ../configs/rcnn_rf_kmeans.yaml --model_name rcnn-bank_ft0.pkl --situation BANK
python ft0_user_selector.py --config_file ../configs/rcnn_rf_kmeans.yaml --model_name rcnn-it_ft0.pkl --situation IT

python ft0_user_selector.py --config_file ../configs/rcnn_rf_kmeans.yaml --model_name rcnn-accom_ft0.pkl --situation ACCOM
python ft0_user_selector.py --config_file ../configs/rcnn_rf_kmeans.yaml --model_name rcnn-wait_ft0.pkl --situation WAIT


python ft0_user_selector.py --config_file ../configs/mobi_rf_kmeans.yaml --model_name mobi-bank_ft0.pkl --situation BANK
python ft0_user_selector.py --config_file ../configs/mobi_rf_kmeans.yaml --model_name mobi-it_ft0.pkl --situation IT

python ft0_user_selector.py --config_file ../configs/mobi_rf_kmeans.yaml --model_name mobi-accom_ft0.pkl --situation ACCOM
python ft0_user_selector.py --config_file ../configs/mobi_rf_kmeans.yaml --model_name mobi-wait_ft0.pkl --situation WAIT
