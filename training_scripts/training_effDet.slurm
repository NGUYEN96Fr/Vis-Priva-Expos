#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition gpu
#SBATCH -w node25
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH -J efDet_v0
#SBATCH --output=/home/users/vnguyen/effDet.out
#SBATCH --error=/home/users/vnguyen/effDet.err

conda activate efDetpytorch
module load cuda/10.2 gcc/7.3
cd /home/users/vnguyen/intern20/Vis-Priva-Expos/models/EfficientDet.pytorch/
python train.py -c 4 -p custom_v0 --batch_size 8 --lr 1e-5 --num_epochs 150 \
 --load_weights /home/users/vnguyen/intern20/PRETRAINED/effDetTorch/efficientdet-d4.pth --data_path /home/users/vnguyen/intern20/DATA/CUSTOM/V0 \
 --saved_path /home/users/vnguyen/intern20/OUTPUTS/effDet/custom_v0