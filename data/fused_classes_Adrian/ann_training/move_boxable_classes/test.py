import os
import tarfile
import cv2

ann_path = '/home/vankhoa/DATA/IMNET_Sources/Annotation/n03360622/n03360622_4283.xml'
class_tar = '/scratch_global/DATASETS/ImageNet/tars/n03360622.tar'

df = tarfile.open(class_tar)
df.extractall(path= '/scratch_global/vankhoa/IMAGENET')

img_path =  '/scratch_global/vankhoa/IMAGENET/n03360622_4283.JPEG'

img = cv2.imread(img_path)

print(img.shape)

import xml.etree.ElementTree as ET

tree = ET.parse(ann_path)
root = tree.getroot()

img_width = root[3][0].text
img_heigh = root[3][1].text

print(img_heigh,img_width)

for object in root.iter('object'):
    for bbox in object.iter('bndbox'):
        print(bbox[0].text,bbox[1].text,bbox[2].text, bbox[3].text)
