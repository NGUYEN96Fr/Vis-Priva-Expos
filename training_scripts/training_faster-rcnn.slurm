#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition gpu-test
#SBATCH -w node21
#SBATCH -J Retinanet_v0
#SBATCH --gres=gpu:2

#SBATCH --output=/home/users/vnguyen/retinanet_v0.out
#SBATCH --error=/home/users/vnguyen/retinanet_v0.err

conda activate detectron2
module load cuda/10.2 gcc/7.3
cd /home/users/vnguyen/intern20/Vis-Priva-Expos/models/detectron2
python retinanet_custom.py --config-file configs/retinanet.yaml --num-gpus 2