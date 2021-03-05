ssh node21
conda load cuda/9.2 gcc/7.3
conda activate detectron2 # cet environement du anaconda doit contenir pytorch et cuda
git clone https://github.com/facebookresearch/detectron2.git # telecharger detectron2
python3 -m pip install -e detectron2
