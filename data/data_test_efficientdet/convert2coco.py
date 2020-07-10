"""
This file convert annotated images into the coco format.

"""
import os
from datetime import date
import yaml
import shutil
import cv2
import json
import tqdm



class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def _generate_info(params):
    """
    Generate info for the custom data-set

    :return: dict
    """
    print('Generating info ...')
    info = {'description': params.description, 'url': params.url, 'version': params.version, 'contributor': params.contributor,
            'date_created': date.today().strftime('%Y-%m-%d')}

    return info


def _generate_licences():
    """

    :return: list
    """
    print('Generating licences ...')
    return [{'url': None, 'id': 1, 'name': None}]


def _generate_categories(params):
    """

    :param path: string
        path to a .txt formatted file containing categories

    :return: list
    """
    print('Generating categories ...')
    with open(params.category_path) as fp:
        id = 1
        category_list = []
        categories = fp.readlines()
        for category in categories:
            # indexed from the 1-based index system
            category_list.append({'supercategory': 'None', 'id': id, 'name': category.replace('\n','')})
            id +=1

    return category_list


def _generate_images(source, image_paths, saved_path):
    """
    Generate a dictionary containing image info. The method will depend on where
    images come from.

    :param img_path: string
        path to images
    :param saved_path : string
        path to saved images
    :param source: string
        source image

    :return: list
    """
    print('Generating images ...')
    if not os.path.exists(params.saved_path):
        os.mkdir(params.saved_path)

    img_files =[]
    if source == 'imt':
        files = os.listdir(image_paths)
        img_files = [file for file in files if file.split('.')[-1] != "txt"]

    img_list = []
    img_id_mapping = {}
    img_id = 0
    for img_file in tqdm.tqdm(img_files):

        # get img_path, img's width & height
        img_path = os.path.join(image_paths,img_file)
        img = cv2.imread(img_path)
        height, width, _ = img.shape

        # append to the list, mapping, and cp to the saved directory
        img_list.append({'license': 1, 'file_name': img_file, 'coco_url': '', 'height': height, 'width': width, 'date_captured': '', 'flickr_url': '', 'id': img_id})
        img_id_mapping[img_file.split('.')[0]] = {'id': img_id, 'name': img_file, 'height': height, 'width': width}
        shutil.copy(img_path,saved_path)
        img_id += 1

    print('--- Saving ...')

    with open('img_list.json', 'w') as f0:
        json.dump(img_list, f0)

    with open('img_id_mapping.json', 'w') as f1:
        json.dump(img_id_mapping, f1)

    return img_list, img_id_mapping


def _generate_annotations(source, ann_paths, img_id_mapping):
    """
    Generate a dictionary containing annotation info. The method will depend on where annotations
    come from.

    :param ann_paths: string
        path to annotation files

    :param img_id_mapping : dict
        dict mapping between img file name and it id in  the CoCo format

    :param source: string
        source

    :return: list
    """
    print('Generating annotations ...')
    anno_files = []
    if source == 'imt':
        files = os.listdir(ann_paths)
        anno_files = [file for file in files if file.split('.')[-1] == 'txt']

    anno_list = []
    id_anno = 0
    for anno_file in tqdm.tqdm(anno_files):

        if source == 'imt':
            # get the corresponding image
            img_meta = img_id_mapping[anno_file.split('.')[0]]
            img_file, img_id, width, height = img_meta['name'], img_meta['id'], img_meta['width'], img_meta['height']

            #get annotation info
            anno_path = os.path.join(ann_paths,anno_file)

            with open(anno_path) as fp:
                bboxes = fp.readlines()
                for bbox_line in bboxes:
                    #use the 1 - based index system for categories
                    bbox_meta = bbox_line.split(' ')
                    cat_id = int(bbox_meta[0]) + 1
                    x_top_left = int((float(bbox_meta[1]) - float(bbox_meta[3])/2)*width)
                    y_top_left = int((float(bbox_meta[2])-float(bbox_meta[4])/2)*height)
                    w_new = int(float(bbox_meta[3])*width)
                    h_new = int(float(bbox_meta[4])*height)
                    anno_list.append({'segmentation': [], 'area': None, 'iscrowd': 0, 'image_id': img_id,\
                                      'bbox': [x_top_left, y_top_left, w_new, h_new], 'category_id': cat_id, 'id': id_anno})
                    id_anno += 1

    return anno_list


def convert2coco(params):
    """

    :param params: Params

    :return: dict
    """
    load = False

    info = _generate_info(params)
    licenses = _generate_licences()
    category_list = _generate_categories(params)

    if not load:
        img_list, img_id_mapping = _generate_images(params.source,params.image_paths,params.saved_path)

    else:
        print('Loading generated images ...')

        with open('img_list.json', 'w') as f3:
            img_list = json.loads(f3.read())
        with open('img_id_mapping.json', 'w') as f4:
            img_id_mapping = json.loads(f4.read())

    anno_list = _generate_annotations(params.source, params.ann_paths, img_id_mapping)
    coco_formatted_dataset = {'info': info, 'licenses': licenses, 'images': img_list, 'annotations': anno_list, 'categories': category_list}

    with open(params.saved_data_name, 'w') as f5:
        json.dump(coco_formatted_dataset, f5)

    print('Finish !')
if __name__ == '__main__':
    params = Params('./config.yml')
    convert2coco(params)