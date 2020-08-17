"""
This module is used to search classes on three databases COCO, OpenImage, ImageNet
"""
import os
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
         {'a,b,c' = [0,2,5], ...}

    """
    anno_path = '/scratch_global/vankhoa/coco14-annotations'
    train_path = os.path.join(anno_path, 'instances_train2014.json')
    val_path = os.path.join(anno_path, 'instances_val2014.json')

    coco_stats = {}
    coco_train = COCO(train_path)
    coco_val = COCO(val_path)

    cats = coco_train.loadCats(coco_train.getCatIds())
    nms = [cat['name'] for cat in cats]
    print(type(nms))
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    for class_ in list_classes:

        if len(class_.split(',')) > 1:

            if class_ not in coco_stats:
                coco_stats[class_] = [0 for i in range(len(class_.split(',')))]

            for i, sub_class in enumerate(class_.split(',')):

                cat_id = coco_train.getCatIds(catNms=sub_class)

                if len(cat_id) != 0:
                    id_number = cat_id[0]

                    if nms[id_number-1] == sub_class:
                        img_train_ids = coco_train.getImgIds(catIds=cat_id)
                        img_val_ids = coco_val.getImgIds(catIds=cat_id)

                    else:
                        img_train_ids = []
                        img_val_ids = []
                else:
                    img_train_ids = []
                    img_val_ids = []

                coco_stats[class_][i] += len(img_train_ids)
                coco_stats[class_][i] += len(img_val_ids)

        else:

            if class_ not in coco_stats:
                coco_stats[class_] = 0

            cat_id = coco_train.getCatIds(catNms=class_)

            if len(cat_id) != 0:
                id_number = cat_id[0]

                if nms[id_number - 1] == class_:
                    img_train_ids = coco_train.getImgIds(catIds=cat_id)
                    img_val_ids = coco_val.getImgIds(catIds=cat_id)

                else:
                    img_train_ids = []
                    img_val_ids = []
            else:
                img_train_ids = []
                img_val_ids = []

            coco_stats[class_] += len(img_train_ids)
            coco_stats[class_] += len(img_val_ids)

    print(coco_stats)

    return coco_stats


def search():
    """

    Returns
    -------

    """
    file_name = 'fused_classes_v0.txt'
    list_classes = read_file(file_name)
    ## MS COCO 2014 search
    coco_stats = ms_coco_search(list_classes)


if __name__ == '__main__':
    search()
