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


def _generate_categories_anns_images(ann_file, class_save_file):
    """
    Generate dictionaries containing category, image and annotation infos

    :param ann_file: string
        annotation file
    :param img_path : string
        path to image directories
    :param save_path: string
        saved image path

    :return: list
    """
    # Category Info
    cat_coco_list = []
    cat_id_mapping = {}
    cat_id = 1

    # Image Info
    cats_img_files = {} # categories and its image files
    img_coco_list = []
    image_id = 0
    image_id_mapping = {}

    # Annotation Info
    cats_bbox_sizes = {} # categories and bounding box sizes
    ann_coco_list = []
    ann_id = 0


    print('Generating coco category, image and ann info ...')

    ##Reading the annotation file.
    with open(ann_file) as fp:
        ann_lines = fp.readlines()

        for ann in tqdm.tqdm(ann_lines):
            parts = ann.split('\t')
            if parts[1] != 'skipped':

                img_size = parts[1].split('x')
                width, height = int(img_size[0]), int(img_size[1])

                ## CATEGORY
                category =  parts[2]
                if category not in cat_id_mapping:
                    cat_id_mapping[category] = cat_id
                    cat_coco_list.append({'supercategory': 'None', 'id': cat_id, 'name': category.replace('_',' ')})
                    cat_id += 1

                if category not in cats_img_files:
                    cats_img_files[category] = []
                    cats_bbox_sizes[category] = []
                ## IMAGE
                img_file = parts[0]
                if img_file not in cats_img_files[category]:
                    cats_img_files[category].append(img_file)
                    ## add the image to the image coco list
                    img_coco_list.append(
                        {'license': 1, 'file_name': img_file, 'coco_url': '', 'height': height, 'width': width,
                         'date_captured': '', 'flickr_url': '', 'id': image_id})
                    image_id_mapping[img_file] = image_id
                    image_id +=1

                ## ANNOTATION
                x1,y1,x2,y2 = [float(coord) if float(coord) > 0 else 0 for coord in parts[3].split(' ')]
                w_bbox = x2 - x1
                h_bbox = y2 - y1

                if w_bbox > 0 and h_bbox > 0:
                    ann_coco_list.append({'segmentation': [], 'area': None, 'iscrowd': 0, 'image_id': image_id_mapping[img_file], \
                                      'bbox': [x1, y1, w_bbox, h_bbox], 'category_id': cat_id_mapping[category],
                                      'id': ann_id})
                    cats_bbox_sizes[category].append((w_bbox,h_bbox))
                    ann_id += 1


    # Generate class files
    writer = open(class_save_file, 'w')
    writer.write('%s \n'%cat_id_mapping)
    for cate in cat_id_mapping:
        writer.write('%s, '%cate)
    writer.close()


    return cat_coco_list, img_coco_list, ann_coco_list

def convert2coco(params):
    """"""

    info = _generate_info(params)
    licenses = _generate_licences()
    category_list, img_list, anno_list = _generate_categories_anns_images(params.ann_file,\
                                                                          params.class_save_file)

    coco_formatted_dataset = {'info': info, 'licenses': licenses, 'images': img_list, 'annotations': anno_list,
                              'categories': category_list}

    with open(params.saved_data_name, 'w') as f5:
        json.dump(coco_formatted_dataset, f5)

    print('Finish !')

if __name__ == '__main__':
    params = Params('./config.yaml')
    convert2coco(params)