#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --partition gpup100
#SBATCH -w node29
#SBATCH --gres=gpu:1 env | grep CUDA_VISIBLE_DEVICES

#SBATCH --output=/home/users/vnguyen/detec2.out
#SBATCH --error=/home/users/vnguyen/detec2.err

module load cuda/10.2 gcc/7.3

cd /home/users/vnguyen/intern20/Vis-Priva-Expos/models/EfficientDet.pytorch/

conda activate efDetpytorch

python train.py -c 0 -p custom30cls --batch_size 8 --lr 1e-5 --num_epochs 10 \
 --load_weights /home/users/vnguyen/intern20/Pretrained-Weight/effDetTorch/efficientdet-d0.pth --data_path /home/users/vnguyen/intern20/DATA/250_imgs_per_class_cluster_IMT/ \
 --saved_path /home/users/vnguyen/intern20/OUTPUTS/effDet