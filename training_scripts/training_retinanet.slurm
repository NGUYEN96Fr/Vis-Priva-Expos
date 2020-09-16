#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition gpup100
#SBATCH -w node29
#SBATCH --gres=gpu:1 env | grep CUDA_VISIBLE_DEVICES

#SBATCH --output=/home/users/vnguyen/detec2.out
#SBATCH --error=/home/users/vnguyen/detec2.err

module load cuda/10.2 gcc/7.3
cd /home/users/vnguyen/intern20/Vis-Priva-Expos/models/detectron2
python custom_train.py