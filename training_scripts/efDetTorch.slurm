#!/bin/bash
#SBATCH -N 3
#SBATCH -n 4
#SBATCH --partition gpu
#SBATCH -w node23,node24,node25
#SBATCH --gres=gpu:4 env | grep CUDA_VISIBLE_DEVICES

#SBATCH --output=/home/users/vnguyen/efdettorch.out
#SBATCH --error=/home/users/vnguyen/efdettorch.errs

module load /cuda/9.2
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
echo "CUDA VISIBLE DEVICES: $CUDA_VISIBLE_DEVICES"
cd /home/users/vnguyen/intern20/Vis-Priva-Expos/models/EfficientDet.pytorch/
python train.py -c 0 -p shape --data_path /home/users/vnguyen/intern20/DATA/effDet --head_only True --lr 1e-3 --batch_size 32 --load_weights /home/users/vnguyen/intern20/Pretrained-Weight/effDetTorch/efficientdet-d0.pth  --num_epochs 50