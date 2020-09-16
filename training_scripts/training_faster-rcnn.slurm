#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition gpu
#SBATCH -w node18
#SBATCH -J Faster-RCNN_v0
#SBATCH --gres=gpu:8
#SBATCH --exclusive

#SBATCH --output=/home/users/vnguyen/faster-rcnn_v0.out
#SBATCH --error=/home/users/vnguyen/faster-rcnn_v0.err

conda activate detectron2
module load cuda/9.2 gcc/7.3 cmake/3.15.3
cd /home/users/vnguyen/intern20/Vis-Priva-Expos/models/detectron2
python -m pip install -e detectron2
python faster-rcnn_custom.py --config-file configs/faster-rcnn.yaml --num-gpus 8