"""
This file:
 - retrieves bounding boxes of each classes in the detection datasets
 - retrieves corresponding images
 - arranges classes into detection model training data, and annotating model training data

"""
import os
import json
import csv
import shutil
import skimage.io as io
import cv2
from pycocotools.coco import COCO


# txt_main_path = '/scratch_global/vankhoa/official_train_inference_ann_models/inference'
img_infer_train_save_dir = '/scratch_global/vankhoa/official_train_inference_ann_models/images'
img_dect_train_dir = '/scratch_global/vankhoa/official_train_inference_ann_models/dect_training'
img_ann_train_dir = '/scratch_global/vankhoa/official_train_inference_ann_models/ann_training'
anno_dect_train_dir = '/scratch_global/vankhoa/official_train_inference_ann_models/dect_training/annotations'
anno_ann_train_dir = '/scratch_global/vankhoa/official_train_inference_ann_models/ann_training/annotations'
N_max = 3000 # number of max images per class

def load():
    """
    detection training classes with sufficient images ( > 2000 images)

    Returns
    -------

    """
    suff_dect = {} # classes have sufficient boxed images to train detection models > 2000
    suff_ann = {} # classes have sufficient boxed images to train annotating models   > 250 and < 2000

    with open('task_2-1_suff_boxable_detection.txt') as fp:
        lines = fp.readlines()
        for line in lines:
            parts = line.split('\\')
            suff_dect[parts[0]] = int(parts[1])

    with open('task_1_suff_boxable_classes.txt') as fp:
        lines = fp.readlines()
        for line in lines:
            parts = line.split('\\')
            suff_ann[parts[0]] = int(parts[1])

    return suff_dect, suff_ann

def check_return(file_path):
    """
    Check of a current file exist, and return its content as a list, and its new writer

    Returns
    -------

    """
    if not os.path.exists(file_path):
        writer = open(file_path, 'w')
        content = []

    else:
        writer = open(file_path, 'a')

        content = []
        with open(file_path) as fp:
            lines = fp.readlines()
            for line in lines:
                content.append(line.split('\n')[0])

    return writer, content



def retrieve_coco(suff_dect, suff_ann):
    """

    Returns
    -------
     - load current image and bounding box containers and do statistics
     - check if nb of images >= N_max
         - add new images and bounding box to pre-defined folder
         - add to statistic files

    """

    anno_path = '/home/vankhoa/DATA/CoCo_2014/annotations'
    train_path = os.path.join(anno_path, 'instances_train2014.json')
    val_path = os.path.join(anno_path, 'instances_val2014.json')

    coco_train = COCO(train_path)
    coco_val = COCO(val_path)

    with open(train_path, 'r') as fp:
        js = json.loads(fp.read())
        categories = js['categories']

    coco_cats = {}
    for cat in categories:
        coco_cats[cat['name']] = cat['id']

    with open('coco_imgs.txt', 'r') as fp:
        coco_imgs = json.loads(fp.read())

    def write_anno_img(img_id, category, bbox_writer, bbox_content, img_writer, img_content):
        """
        check annotation of a current image with a current categories
        write its to the image container, and the bounding box container

        Returns
        -------

        """

        # check if img_id in train or test datasets, download the image, and
        # write to the corresponding bounding box file
        try:
            ann_train_ids = coco_train.getAnnIds(imgIds=img_id, catIds=coco_cats[category], iscrowd=None)
        except:
            ann_train_ids = []

        try:
            ann_val_ids = coco_val.getAnnIds(imgIds=img_id, catIds=coco_cats[category], iscrowd=None)
        except:
            ann_val_ids = []

        if len(ann_train_ids) > 0:
            img = coco_train.loadImgs(img_id)[0]
        if len(ann_val_ids) > 0:
            img = coco_val.loadImgs(img_id)[0]
        if len(ann_train_ids) > 0 and len(ann_val_ids) > 0:
            raise ValueError('Co-exist error !!!')

        img_name = img['file_name']
        img_url = img['coco_url']
        img_width = img['width']
        img_height = img['height']

        img_dst = os.path.join(img_infer_train_save_dir, img_name)

        if img_dst not in img_content:
            # move the image, add to the current list, write to the current writer
            img_content.append(img_dst)

        if len(img_content) <= N_max:
            img_src = io.imread(img_url)
            io.imsave(img_dst,img_src)
            img_writer.write('%s\n' % img_dst)

            if len(ann_train_ids) > 0:
                for id in ann_train_ids:
                    anno = coco_train.loadAnns(id)[0]
                    bbox = anno['bbox']
                    bbox_wrt_info = '%s\\%s\\%s\\%s\\%s\\%s\\%s' % (
                        img_name, img_width, img_height, bbox[0], bbox[1], bbox[2], bbox[3])

                    if bbox_wrt_info not in bbox_content:
                        bbox_writer.write('%s\n' % bbox_wrt_info)
                        bbox_content.append(bbox_wrt_info)

            if len(ann_val_ids) > 0:
                for id in ann_val_ids:
                    anno = coco_val.loadAnns(id)[0]
                    bbox = anno['bbox']
                    bbox_wrt_info = '%s\\%s\\%s\\%s\\%s\\%s\\%s' % (
                        img_name, img_width, img_height, bbox[0], bbox[1], bbox[2], bbox[3])

                    if bbox_wrt_info not in bbox_content:
                        bbox_writer.write('%s\n' % bbox_wrt_info)
                        bbox_content.append(bbox_wrt_info)

        return bbox_writer, bbox_content,  img_writer, img_content


    def move_(anno_train_dir, img_train_dir, img_ids, class_):
        """
        move to the corresponding type of data

        Returns
        -------

        """
        if len(img_ids) > 0:

            if len(class_.split(',')) > 1:
                # use the first name of class synonyms representing the written files
                first_syn = class_.split(',')[0]
                bbox_writer_path = os.path.join(anno_train_dir, first_syn+ '.txt')
                img_writer_path = os.path.join(img_train_dir, first_syn+ '.txt')

                bbox_writer, bbox_content = check_return(bbox_writer_path)
                img_writer, img_content = check_return(img_writer_path)

                for sub_class in class_.split(','):
                    if sub_class in coco_cats:
                        for img_id in img_ids:
                            bbox_writer, bbox_content,  img_writer, img_content = write_anno_img(int(img_id), sub_class,
                                                                     bbox_writer, bbox_content, img_writer, img_content)

                print(len(img_content),' images')

                bbox_writer.close()
                img_writer.close()

            else:

                bbox_writer_path = os.path.join(anno_train_dir, class_ + '.txt')
                img_writer_path = os.path.join(img_train_dir, class_ + '.txt')

                bbox_writer, bbox_content = check_return(bbox_writer_path)
                img_writer, img_content = check_return(img_writer_path)

                for img_id in img_ids:
                    bbox_writer, bbox_content,  img_writer, img_content = write_anno_img(int(img_id), class_,
                                                             bbox_writer, bbox_content, img_writer, img_content)

                print(len(img_content), ' images')

                bbox_writer.close()
                img_writer.close()


    for class_, img_ids in coco_imgs.items():

        if class_ in suff_dect:
            # move into dect training
            print(class_, ': dect')
            move_(anno_dect_train_dir, img_dect_train_dir, img_ids, class_)


        elif class_ in suff_ann:
            ## move into annotation training
            if len(img_ids) > 0:
                print(class_, ': anno')
                move_(anno_ann_train_dir, img_ann_train_dir, img_ids, class_)


def retrieve_openimg(suff_dect, suff_ann):
    """

    Parameters
    ----------
    suff_dect
    suff_ann

    Returns
    -------

    """
    saved_openimage_bergamote = '/scratch_global/DATASETS/openimages/v4/images/nozip'

    with open('openimage_imgs.txt') as fp:
        openimg_data = json.loads(fp.read())

    ann_path = '/home/vankhoa/Vis-Priva-Expos/data/Files_BERGAMOTE/openimg_anns/'
    train = 'train-annotations-bbox.csv'
    val = 'validation-annotations-bbox.csv'
    test = 'test-annotations-bbox.csv'
    ann_files = [train, val, test]

    img_bbox = {}
    for ann_file in ann_files:
        with open(ann_path + ann_file) as fp:
            reader = csv.reader(fp, delimiter=',')
            skip = True
            for row in reader:
                if skip:
                    skip = False
                else:
                    img_bbox[row[0]+row[2]] = [float(row[4]),float(row[6]),   ## x_left, y_left, bb_width, bb_height
                                               float(row[5]) - float(row[4]), float(row[7]) -float(row[6])]

    def move_(anno_train_dir, img_train_dir, img_ids, class_):
        """

        Parameters
        ----------
        anno_dect_train_dir
        img_dect_train_dir
        img_ids
        class_

        Returns
        -------

        """
        if len(class_.split(';')[0].split(',')) > 1:
            # use the first name of class synonyms representing the written files
            first_syn = class_.split(';')[0].split(',')[0]

        else:
            first_syn = class_.split(';')[0]


        bbox_writer_path = os.path.join(anno_train_dir, first_syn + '.txt')
        img_writer_path = os.path.join(img_train_dir, first_syn + '.txt')

        bbox_writer, bbox_content = check_return(bbox_writer_path)
        img_writer, img_content = check_return(img_writer_path)


        for img in img_ids:
            img_name = img + '.jpg'

            img_dst_dir = os.path.join(img_infer_train_save_dir, img_name)
            img_src_dir = os.path.join(saved_openimage_bergamote, img_name)

            if os.path.exists(img_src_dir):
                if img_dst_dir not in img_content:
                    img_content.append(img_dst_dir)

                if len(img_content) <= N_max:

                    img_ = cv2.imread(img_src_dir)
                    img_height, img_width, _ = img_.shape
                    norm_bbox = img_bbox[img+class_.split(';')[1]]

                    new_xleft, new_yleft = norm_bbox[0]*img_width, norm_bbox[1]*img_height
                    new_bbwidth, new_bbheight = norm_bbox[2]*img_width,  norm_bbox[3]*img_height

                    bbox_wrt_info = '%s\\%s\\%s\\%s\\%s\\%s\\%s' % (
                        img_name, img_width, img_height, new_xleft, new_yleft, new_bbwidth, new_bbheight)

                    if bbox_wrt_info not in bbox_content:
                        bbox_writer.write('%s\n'%bbox_wrt_info)
                        bbox_content.append(bbox_wrt_info)

                    print(img_dst_dir)
                    img_writer.write('%s\n'%img_dst_dir)
                    shutil.copy(img_src_dir, img_dst_dir)

        print(len(img,' images'))
        bbox_writer.close()
        img_writer.close()

    for class_, img_ids in openimg_data.items():

        if len(img_ids) > 0:

            if class_.split(';')[0] in suff_dect:
                print(class_.split(';')[0], ': dect')
                move_(anno_dect_train_dir, img_dect_train_dir, img_ids, class_)

            elif class_.split(';')[0] in suff_ann:
                if len(img_ids) > 0:
                    print(class_.split(';')[0], ': anno')
                    move_(anno_ann_train_dir, img_ann_train_dir, img_ids, class_)


def main():
    suff_dect, suff_ann = load()
    print('COCO')
    #retrieve_coco(suff_dect, suff_ann)
    print('OpenImage')
    retrieve_openimg(suff_dect, suff_ann)


if __name__ == '__main__':
    main()
