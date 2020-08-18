"""
This module is used to search classes on three databases COCO, OpenImage, ImageNet
"""
import os
import csv
import tqdm
import json
import numpy as np
from pycocotools.coco import COCO

def read_file(file_name):
    """

    Returns
    -------
    a list of classes

    """
    list_classes = []
    with open(file_name) as fp:
        lines = fp.readlines()
        for line in lines:
            parts = line.split('\n')
            class_ = parts[0]
            list_classes.append(class_)

    return list_classes



def ms_coco_search(list_classes):
    """
        MS COCO 2014
    Parameters
    ----------
    list_classes: list
        classes needed to search in databases

    Returns
    -------
        coco_stats : dict
            nb of images per class

        coco_imgs : dict
            image's ids per class

    """
    anno_path = '/home/vankhoa/DATA/CoCo_2014/annotations'
    train_path = os.path.join(anno_path, 'instances_train2014.json')
    val_path = os.path.join(anno_path, 'instances_val2014.json')

    with open(train_path, 'r') as fp:
        js = json.loads(fp.read())
        categories = js['categories']

    coco_cats = {}
    for cat in categories:
        coco_cats[cat['id']] = cat['name']

    coco_stats = {}
    coco_imgs = {}
    coco_train = COCO(train_path)
    coco_val = COCO(val_path)

    for class_ in tqdm.tqdm(list_classes):

        if len(class_.split(',')) > 1:
            if class_ not in coco_stats:
                coco_stats[class_] = 0
                coco_imgs[class_] = []

            for sub_class in class_.split(','):
                cat_id = coco_train.getCatIds(catNms=sub_class)
                if len(cat_id) != 0:
                    id_number = cat_id[0]
                    if coco_cats[id_number] == sub_class:
                        img_train_ids = coco_train.getImgIds(catIds=id_number)
                        for img in img_train_ids:
                            coco_imgs[class_].append(img)

                        img_val_ids = coco_val.getImgIds(catIds=id_number)
                        for img in img_val_ids:
                            coco_imgs[class_].append(img)

                    else:
                        print(coco_cats[id_number],' ---- ',sub_class)
                        img_train_ids = []
                        img_val_ids = []
                else:
                    img_train_ids = []
                    img_val_ids = []

                coco_stats[class_] += len(img_train_ids)
                coco_stats[class_] += len(img_val_ids)

        else:

            if class_ not in coco_stats:
                coco_stats[class_] = 0
                coco_imgs[class_] =[]

            cat_id = coco_train.getCatIds(catNms=class_)

            if len(cat_id) != 0:
                id_number = cat_id[0]
                if coco_cats[id_number] == class_:
                    img_train_ids = coco_train.getImgIds(catIds=id_number)
                    for img in img_train_ids:
                        coco_imgs[class_].append(img)

                    img_val_ids = coco_val.getImgIds(catIds=id_number)
                    for img in img_val_ids:
                        coco_imgs[class_].append(img)

                else:
                    print(coco_cats[id_number], ' ---- ', class_)
                    img_train_ids = []
                    img_val_ids = []
            else:
                img_train_ids = []
                img_val_ids = []

            coco_stats[class_] += len(img_train_ids)
            coco_stats[class_] += len(img_val_ids)

    print(coco_stats)
    return coco_stats, coco_imgs


def openimage(list_classes):
    """
        OpenImage V4

    Parameters
    ----------
    file_name:
    Returns
    -------
        openimage_stats: dict
            open image v4 statistics

        openimage_imgs: dict
            img's names per class

    """
    openimage_stats = {}
    openimage_imgs = {}
    ann_path = '/home/vankhoa/Vis-Priva-Expos/data/Files_BERGAMOTE/openimg_anns/'
    train = 'train-annotations-bbox.csv'
    val = 'validation-annotations-bbox.csv'
    test = 'test-annotations-bbox.csv'
    ann_files = [train, val, test]

    ## get openimage class names and its encoded names

    op_cls_names = {}
    with open(ann_path + 'class-descriptions-boxable.csv') as fp:
        reader = csv.reader(fp, delimiter=',')
        for row in reader:
            op_cls_names[row[1].split(" (")[0].lower()] = row[0]

    encoded_classes = {}
    img_list = {}
    for class_ in list_classes:
        if len(class_.split(',')) > 1:
            encoded_classes[class_] = ['nex' for i in range(len(class_.split(',')))]
            for i, sub_cls in enumerate(class_.split(',')):
                if sub_cls in op_cls_names:
                    encoded_classes[class_][i] = op_cls_names[sub_cls]
                    img_list[op_cls_names[sub_cls]] = []

        else:
            encoded_classes[class_] = ['nex']  # not exist
            if class_ in op_cls_names:
                encoded_classes[class_] = [op_cls_names[class_]]
                img_list[op_cls_names[class_]] = []

    for ann_file in ann_files:
        with open(ann_path + ann_file) as fp:
            reader = csv.reader(fp, delimiter=',')
            for row in reader:
                if row[2] in img_list:
                    img_list[row[2]].append(row[0])

    for class_, encoded_names in encoded_classes.items():
        if len(encoded_classes) > 1:
            openimage_stats[class_] = 0
            openimage_imgs[class_] = []
            for sub_cls in encoded_names:
                if sub_cls != 'nex':
                    for img in img_list[sub_cls]:
                        openimage_imgs[class_].append(img)
                    openimage_stats[class_] += len(img_list[sub_cls])
        else:
            openimage_stats[class_] = 0
            openimage_imgs[class_] = []
            if encoded_names[0] != 'nex':
                for img in img_list[encoded_names[0]]:
                    openimage_imgs[class_].append(img)
                openimage_stats[class_] = len(img_list[encoded_names[0]])

    return openimage_stats, openimage_imgs

def imnet(list_classes):
    """
        ImageNet

    Parameters
    ----------
    list_classes

    Returns
    -------
        imnet_stats: dict

    """
    class_name_file = '/home/vankhoa/DATA/IMNET_Sources/words.txt'
    ann_path = '/home/vankhoa/DATA/IMNET_Sources/Annotation'

    imnet_stats = {}

    encoded_imnet_classes = {}

    with open(class_name_file) as fp:
        lines = fp.readlines()

        for line in lines:
            parts = line.split('\t')
            id_class = parts[0]
            syn_names = parts[1].split(',')
            encoded_imnet_classes[id_class] = []
            for syn_name in syn_names:
                syn_name = syn_name.split('\n')[0].lstrip().lower()
                encoded_imnet_classes[id_class].append(syn_name)

    excluded_IDs = ['n00521562','n00893088','n00972319','n06648046',
                    'n06879180','n01177033','n06228549','n08087570',
                    'n08088472','n08147188','n08475929','n09641578',
                    'n10018021','n00515069','n00590806','n00591006',
                    'n01814755','n01826680','n01826844','n02284224',
                    'n02766534','n02780315','n03096960','n03150232',
                    'n03209477','n03418052','n03877229','n04098399',
                     'n04101860', 'n04102162','n09679316','n09994673',
                    'n04102285', 'n04102406', 'n04102760',
                    'n04310157', 'n06781383', 'n07349532', 'n08142801',
                    'n09761403', 'n09793946', 'n09896401', 'n09950150',
                    'n09950318', 'n10175418', 'n10181026', 'n10226556',
                    'n10407552', 'n10421470', 'n10525134',
                    'n10553235', 'n10635625', 'n13979786', 'n08351777',
                    'n01847253', 'n02040505', 'n02061853', 'n03215930',
                    'n03216080', 'n03216402', 'n07466415','n00195569',
                    'n00350380', 'n00426928', 'n00972521', 'n00975270',
                    'n01004072', 'n02049088', 'n02081927', 'n03215337',
                    'n04735711', 'n04749709', 'n04751305', 'n04751652',
                    'n05797177', 'n05863302', 'n07042586', 'n07366289',
                    'n10019406', 'n10019733', 'n10410531', 'n10544748',
                    'n10607291', 'n10608188', 'n11694469', 'n11966385',
                    'n12403075', 'n12767208', 'n14032480', 'n14295829',
                    'n14295986', 'n14341923', 'n14557573', 'n14575180',
                    'n07918193','n04579145 ', 'n07918454', 'n07918601']

    encoded_classes = {}
    for class_ in list_classes:
        if len(class_.split(',')) > 1:

            encoded_classes[class_] = [[] for i in range(len(class_.split(',')))]
            for i, sub_class in enumerate(class_.split(',')):
                # check
                for id, classes in encoded_imnet_classes.items():
                    for class_imnet in classes:
                        if sub_class in class_imnet:
                            if id not in excluded_IDs:
                                add_flag = True
                                for k in range(len(class_.split(','))):
                                    if id in encoded_classes[class_][k]:
                                        add_flag = False
                                if add_flag:
                                    encoded_classes[class_][i].append(id)
        else:
            encoded_classes[class_] = []
            if class_ in encoded_imnet_classes:
                for id, classes in encoded_imnet_classes.items():
                    for class_imnet in classes:
                        if class_ in class_imnet:
                            if id not in encoded_classes[class_] and id not in excluded_IDs:
                                encoded_classes[class_].append(id)

    imnet_imgs = {} # image list for each class
    for class_name, ids in encoded_classes.items():
        imnet_imgs[class_name] = []
        if len(ids) > 1:
            for syn_ids in ids:
                for id in syn_ids:
                    if os.path.exists(os.path.join(ann_path,id)):
                        print(class_name + '++++++')
                        imgs = os.listdir(os.path.join(ann_path,id))
                        for img in imgs:
                            imnet_imgs[class_name].append(img)
                    else:
                        print(class_name+'-----')
        else:
            for id in ids:
                if os.path.exists(os.path.join(ann_path, id)):
                    print(class_name + '++++++')
                    imgs = os.listdir(os.path.join(ann_path, id))
                    for img in imgs:
                        imnet_imgs[class_name].append(img)
                else:
                    print(class_name+'-----')

    for class_name, imgs in imnet_imgs.items():
        imnet_stats[class_name] = len(imgs)

    return imnet_stats, imnet_imgs




def search():
    """

    Returns
    -------

    """
    file_name = 'fused_classes_v0.txt'
    list_classes = read_file(file_name)
    ## MS COCO 2014 search
    print('MSCOCO2014')
    coco_stats, coco_imgs = ms_coco_search(list_classes)

    ## Open Image V4
    print('OPENIMAGEV4')
    openimage_stats, openimage_imgs = openimage(list_classes)

    ## ImageNet
    print('IMNET')
    imnet_stats, imnet_imgs = imnet(list_classes)

    print('WRITING')
    sum_stats = []
    class_stats = []
    for class_, nb_img_coco in coco_stats.items():
        class_stats.append(class_)
        sum_stats.append(nb_img_coco + openimage_stats[class_] + imnet_stats[class_])

    sorted_arg_max_min_sum = list(np.argsort(sum_stats)[::-1])
    writer = open('fused_class_bboxable_stats.txt','w')
    writer.write('class\\MSCOCO2014\\OPENIMAGE_V4\\IMAGENET\SUM\n')
    for index in sorted_arg_max_min_sum:
        writer.write('%s\\%s\\%s\\%s\\%s\n'%(class_stats[index],coco_stats[class_stats[index]],
                                             openimage_stats[class_stats[index]],imnet_stats[class_stats[index]],
                                             sum_stats[index]))

    writer.close()

if __name__ == '__main__':
    search()