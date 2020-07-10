""""
Count number of images in retrieved manual and automatic annotations
"""

import os

selected_class_path = './selected_classes.txt'
auto_path = '/scratch_global/vankhoa/RETRIEVED_IMAGES_ImgeNet_OpenImage_v4/auto/'
manual_path = '/scratch_global/vankhoa/RETRIEVED_IMAGES_ImgeNet_OpenImage_v4/manual/'

# read selected classes
sel_classes = []
with open(selected_class_path) as fp:
    classes = fp.readlines()
    for class_ in classes:
        sel_classes.append(class_.replace('\n', ''))

manual_writer = open('manual_statistics.txt','w')
auto_writer = open('automatic_statistics.txt','w')

## count imgs from auto annotations
auto_classes = os.listdir(auto_path)
auto_dict = {}
for class_ in auto_classes:
    class_path = os.path.join(auto_path,class_)
    class_ = class_.replace('_',' ')
    nb_imgs = len(os.listdir(class_path))
    auto_dict[class_] = nb_imgs

manual_classes = os.listdir(manual_path)
manual_dict = {}
for class_ in manual_classes:
    class_path = os.path.join(manual_path,class_)
    class_ = class_.replace('_',' ')
    nb_imgs = len(os.listdir(class_path))
    manual_dict[class_] = nb_imgs

## count imgs from manual annotations
for class_ in sel_classes:

    if class_ in manual_dict:
        manual_writer.write('%s-%s\n'%(class_,manual_dict[class_]))
    else:
        manual_writer.write('%s-%s\n' % (class_, 0))

    if class_ in auto_dict:
        auto_writer.write('%s-%s\n'%(class_,auto_dict[class_]))
    else:
        auto_writer.write('%s-%s\n' % (class_, 0))

manual_writer.close()
auto_writer.close()