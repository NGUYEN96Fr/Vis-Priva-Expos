"""
This module split the manually annotated data into train and split data-sets

"""

import os
import shutil

img_path = '/home/users/vnguyen/intern20/DATA/CUSTOM'
train_save_path = '/home/users/vnguyen/intern20/DATA/V0/train'
calib_save_path = '/home/users/vnguyen/intern20/DATA/V0/calib'
ann_file = 'nicu_v0.txt'
train_split_file = 'train_v0.txt'
calib_split_file = 'calib_v0.txt'
nb_calib_imgs = 20


calib_imgs = []
train_imgs = []

categories = os.listdir(img_path)
for cate in categories:
    counter = 0
    cate_path = os.path.join(img_path,cate)
    imgs = os.listdir(cate_path)
    for img in imgs:
        img_src = os.path.join(cate_path, img)
        counter += 1
        if counter <= nb_calib_imgs:
            img_dst = os.path.join(calib_save_path,img)
            calib_imgs.append(img)
        else:
            img_dst = os.path.join(train_save_path,img)
            train_imgs.append(img)

        shutil.copy(img_src, img_dst)

train_writer = open(train_split_file, 'w')
calib_writer = open(calib_split_file, 'w')

with open(ann_file) as fp:
    lines = fp.readlines()
    for line in lines:
        img = line.split('\t')[0]
        if img in train_imgs:
            train_writer.write('%s'%line)
        if img in calib_imgs:
            calib_writer.write('%s'%line)

train_writer.close()
calib_writer.close()