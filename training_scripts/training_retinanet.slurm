#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition gpu
#SBATCH -w node26
#SBATCH -J Retinanet_v0
#SBATCH --gres=gpu:8
#SBATCH --exclusive

#SBATCH --output=/home/users/vnguyen/retinanet_v0_1.out
#SBATCH --error=/home/users/vnguyen/retinanet_v0_1.err

conda activate detectron2
module load cuda/9.2 gcc/7.3 cmake/3.15.3
cd /home/users/vnguyen/intern20/Vis-Priva-Expos/models/detectron2
python -m pip install -e detectron2
python retinanet_custom.py --config-file configs/retinanet.yaml --num-gpus 8