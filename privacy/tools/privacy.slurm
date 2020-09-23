#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition gpu-test
#SBATCH -w node21

#SBATCH --output=/home/users/vnguyen/intern20/Vis-Priva-Expos/privacy/tools/privacy.out
#SBATCH --error=/home/users/vnguyen/intern20/Vis-Priva-Expos/privacy/tools/privacy.errs

## python train_model.py --config_file ../configs/rf.yaml --model_name RF1.pkl
## python model_info.py --model_name out/RF1.pkl
python train_cnn.py --config_file ../configs/cnn.yaml --model_name vispel_net.pth