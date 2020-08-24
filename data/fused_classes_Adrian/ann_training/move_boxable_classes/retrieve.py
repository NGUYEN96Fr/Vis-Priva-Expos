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

        if len(img_content) <= N_max:

            img_dst = os.path.join(img_infer_train_save_dir, img_name)

            if img_dst not in img_content:
                # move the image, add to the current list, write to the current writer

                if not os.path.exists(img_dst):
                    img_src = io.imread(img_url)
                    io.imsave(img_dst, img_src)

                img_writer.write('%s\n' % img_dst)
                img_content.append(img_dst)

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
                    concat_img_class = row[0]+row[2]
                    if concat_img_class not in img_bbox:
                        img_bbox[concat_img_class] = []

                    img_bbox[concat_img_class].append([float(row[4]), float(row[6]),   ## x_left, y_left, bb_width, bb_height
                                               float(row[5]) - float(row[4]), float(row[7]) -float(row[6])])

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
                    img_writer.write('%s\n'%img_dst_dir)
                    if not os.path.exists(img_dst_dir):
                        shutil.copy(img_src_dir, img_dst_dir)

                if len(img_content) <= N_max:

                    img_ = cv2.imread(img_src_dir)
                    img_height, img_width, _ = img_.shape
                    norm_bboxes = img_bbox[img+class_.split(';')[1]]

                    for norm_bbox in norm_bboxes:

                        new_xleft, new_yleft = norm_bbox[0]*img_width, norm_bbox[1]*img_height
                        new_bbwidth, new_bbheight = norm_bbox[2]*img_width,  norm_bbox[3]*img_height

                        bbox_wrt_info = '%s\\%s\\%s\\%s\\%s\\%s\\%s' % (
                            img_name, img_width, img_height, new_xleft, new_yleft, new_bbwidth, new_bbheight)

                        if bbox_wrt_info not in bbox_content:
                            bbox_writer.write('%s\n'%bbox_wrt_info)
                            bbox_content.append(bbox_wrt_info)
                else:

                    break

        print(len(img_content), ' images')
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


def retrieve_imnet(suff_dect, suff_ann):
    """

    Parameters
    ----------
    suff_dect
    suff_ann

    Returns
    -------

    """
    import tarfile
    import xml.etree.ElementTree as ET

    anno_dir_path = '/home/vankhoa/DATA/IMNET_Sources/Annotation'
    img_dir_path = '/scratch_global/DATASETS/ImageNet/tars'
    extract_img_path = '/scratch_global/vankhoa/IMAGENET'
    extract_tar = []

    with open('imnet_imgs.txt') as fp:
        imnet_data = json.loads(fp.read())


    def move_(anno_train_dir, img_train_dir, img_ids, class_, extract_tar):
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
        if len(class_.split(',')) > 1:
            # use the first name of class synonyms representing the written files
            first_syn = class_.split(',')[0]
        else:
            first_syn = class_


        bbox_writer_path = os.path.join(anno_train_dir, first_syn + '.txt')
        img_writer_path = os.path.join(img_train_dir, first_syn + '.txt')

        bbox_writer, bbox_content = check_return(bbox_writer_path)
        img_writer, img_content = check_return(img_writer_path)

        for img_id in img_ids:

            class_code = img_id.split('_')[0]
            class_tar = class_code + '.tar'

            # check file exist
            if os.path.exists(os.path.join(img_dir_path,class_tar)):

                if len(img_content) <= N_max:

                    if class_tar not in extract_tar:
                        df = tarfile.open(os.path.join(img_dir_path, class_tar))
                        df.extractall(path=extract_img_path)
                        extract_tar.append(class_tar)

                    img_name = img_id.split('.')[0] + '.JPEG'

                    anno_path = os.path.join(anno_dir_path, class_code, img_id)
                    img_src_path = os.path.join(extract_img_path,img_name)
                    img_dst_path = os.path.join(img_infer_train_save_dir, img_name)

                    if os.path.exists(img_src_path):
                        if os.path.exists(anno_path):

                            if img_dst_path not in img_content:
                                shutil.copy(img_src_path, img_dst_path)
                                img_writer.write('%s\n' % img_dst_path)
                                img_content.append(img_dst_path)

                            tree = ET.parse(anno_path)
                            root = tree.getroot()

                            img_relative_width = float(root[3][0].text)
                            img_relative_heigh = float(root[3][1].text)

                            loaded_img = cv2.imread(img_dst_path)
                            img_heigh, img_width, _ = loaded_img.shape

                            width_factor = img_width/img_relative_width
                            heigh_factor = img_heigh/img_relative_heigh

                            for object in root.iter('object'):
                                for bbox in object.iter('bndbox'):
                                    x_left, y_left, bb_width, bb_heigh = float(bbox[0].text)*width_factor, float(bbox[1].text)*heigh_factor,\
                                                                         (float(bbox[2].text) - float(bbox[0].text))*width_factor, \
                                                                         (float(bbox[3].text) - float(bbox[1].text))*heigh_factor
                                    bbox_wrt_info = '%s\\%s\\%s\\%s\\%s\\%s\\%s' % (
                                        img_name, img_width, img_heigh, x_left, y_left, bb_width, bb_heigh)

                                    if bbox_wrt_info not in bbox_content:
                                        bbox_writer.write('%s\n' % bbox_wrt_info)
                                        bbox_content.append(bbox_wrt_info)

        img_writer.close()
        bbox_writer.close()

        print(len(img_content), ' images')

        return extract_tar

    for class_, img_ids in imnet_data.items():

        if len(img_ids) > 0:

            if class_ in suff_dect:
                print(class_, ': dect')
                extract_tar = move_(anno_dect_train_dir, img_dect_train_dir, img_ids, class_, extract_tar)

            elif class_ in suff_ann:
                print(class_, ': anno')
                extract_tar = move_(anno_ann_train_dir, img_ann_train_dir, img_ids, class_, extract_tar)


def main():
    """

    Returns
    -------

    """
    suff_dect, suff_ann = load()
    print('COCO')
    retrieve_coco(suff_dect, suff_ann)

    print('OpenImage')
    retrieve_openimg(suff_dect, suff_ann)

    print('ImageNet')
    retrieve_imnet(suff_dect, suff_ann)


if __name__ == '__main__':
    main()
