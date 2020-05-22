#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Script for utility functions exploited by other programs

"""

import sys,os
import re
import json
import numpy as np
from numpy.linalg import norm


def bb_intersection_over_union(boxA, boxB):
    """Calculate an IoU

    Parameters
    ----------
    boxA : list
        [x_upper_left,y_upper_left,x_lower_right,y_lower_right]
    boxB : list
        [//]

    Returns
    -------
    float
        IoU
    
    References
    ----------
        https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc


    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def prepro_files(path, is_gt = True):
    """Preprocess images and its bounding boxes.

    Parameters
    ---------
    path : string
        A path to the image directory.
    is_gt : bool
        Is the ground truth file ?

    Returns
    -------
    dict
        Containing the images and its bboxes. 
        The return dict having the following structure:      
            {'img1' : {'category1' : [ [box1], ... ], ...} , ...}

    Notes
    -----
        Object id differences btw `test_set.txt` and `verite_terrain.txt`.
        Synchronize the obj id notation by taking the gt obj ids as refereces.

    Raises
    ------
    Exception
        The file path is not correct.

    """

    if not os.path.exists(path):
        raise Exception('The file path is not correct.')

    imgs_bboxes = {} #an img and its bboxes
    #if is_gt:
    #    id_name = {} #map between object ids and object names

    with open(path) as fp:
        
        lines = fp.readlines()
        
        for line in lines: #read the file line by line.
            img = line.strip().split(" ")
            
            if is_gt:
                img_name, obj_id, obj_name, bboxes = img[0], img[1], img[2:-4],[float(img[k]) for k in range(-4,0)] #img, obj info
            else:
                img_name, obj_id, obj_name, bboxes = img[0], str(int(img[1])-1), img[2:-5],[float(img[k]) for k in range(-5,0)] #take the gt obj ids as refs
            
            if len(obj_name) > 1: #in case of at least 2 words saparated by an espace
                obj_nname = ''
                for k in range(len(obj_name)):
                    obj_nname += obj_name[k]
                    if k != len(obj_name) - 1:
                        obj_nname += ' '
            else:
                obj_nname = obj_name[0]

            if img_name not in imgs_bboxes: #the img name aldready added ?
                imgs_bboxes[img_name] = {}
                imgs_bboxes[img_name][obj_nname] = [] #add the obj to the img 
                imgs_bboxes[img_name][obj_nname].append(bboxes)
            else:
                if obj_nname not in imgs_bboxes[img_name]: #the obj name aldready added ?
                    imgs_bboxes[img_name][obj_nname] = []
                imgs_bboxes[img_name][obj_nname].append(bboxes)

    with open('preprocessed_'+path.split('/')[-1].split('.')[0]+'.json','w') as fp:
        json.dump(imgs_bboxes,fp)

    return imgs_bboxes


def preproc_user_file(path):
    """Preprocess an user visual privacy exposure inference file
    
    Parameters
    ----------
    path : string
        The file path

    Returns
    -------
    dict
        Contain user keys with its image info in the following structure:
            {'user_key1': { 'img_id1': { 'category1': [ ( objness1 , [bbx1] ), ... ], ... }, ...}, ...}


    """

    users_meta = {}

    with open(path) as fp:
        lines = fp.readlines()
        
        for line in lines:
            meta = line.split(" ") #extract meta info

            user_id = meta[0]
            img_id = meta[1]
            cate = meta[2] #category
            objness = float(meta[3]) #objectness score
            bbox = [float(meta[k]) for k in range(4,8)] #predicted bbox

            if user_id not in users_meta: #check if the user aldready existed
                users_meta[user_id] = {}
            
            if img_id not in users_meta[user_id]: #check if the img //
                users_meta[user_id][img_id] = {}

            if cate not in users_meta[user_id][img_id]: #check if the category //
                users_meta[user_id][img_id][cate] = []
            
            users_meta[user_id][img_id][cate].append(( objness, bbox)) #add the object info to the user

    #with open('users_meta.json','w') as fp:
    #   json.dump(users_meta,fp)

    return users_meta


def prepro_obj_score(path, print_ = True):
    """Preprocess the object scores per situation file

    Parameters
    ----------
    path : string
        The file path, whose structure:
        {'situ 1': {'domain 1': { 'labels': {'object 1' : int, ...}, '_is_threat': bool }, ... }}
    
    Returns
    -------
    situ_name : string
        situation name
    
    situ_scores : dict
        Containing situations and its object scores in the following structure
            {'object 1' : (score, domain, is_threat), ...}
            Object scores in the situation [0, ... , 6] ==> [-3, ... ,3]

    Raises
    ------
    Exception
        If two domains have the same object.

    """
    with open(path) as fp:
        situ = json.load(fp)

    situ_name = list(situ.keys())[0]
    domains = situ[situ_name]
    situ_scores = {} #scores per a situation
    if print_:
        print('--------------------------------------')
        print('Situation : ',situ_name)
        print('Repeating objects in different domain')

    for domain, items in domains.items():
        objs = items['labels']
        _is_threat = items['_is_threat']

        for obj, score in objs.items():
            
            if obj not in situ_scores:
                situ_scores[obj] = (score - 4,domain,_is_threat)
            
            else:
                if print_:
                    print('+ ',obj)
                #raise Exception('Object existed !!!')

    with open('./prepro_annotations/preprocessed_%s.json'%situ_name,'w') as fp:
        json.dump(situ_scores,fp)

    return situ_name, situ_scores
    

def  prepro_threshs(path, sav_path, method = 'fdr_indiv'):
    """Prerpocess objectness threshold file

    Parameters
    ----------
    path : string
        the path file
    method : string
        the used method for estimating the objectness threshold (default: indi)
        + fdr_indiv : individual FDR threshold for each object
        + fdr_class_x : a common x FDR value for all class
        + fdr_global_x : a global x thredhold for all classes
    
    sav_path: string
        saved file path

    Returns
    -------
    dict
        processed objness threshs, whose structures
            {'object 1': float, ... }            

    Raises
    ------
        ValueError
            When the method name is not approriate.


    """
    threshs = {}

    if not os.path.exists(sav_path + method):
        os.makedirs(sav_path + method)

    if method.split('_')[1] != 'global':
        
        with open(path) as fp:
            skip_line = True #skip first line
            lines = fp.readlines()

            for line in lines:
                
                if skip_line:
                    skip_line = False
                
                else:
                    obj_thresh_fdr = line.split(" ")
                    obj = obj_thresh_fdr[0]
                    thresh = float(obj_thresh_fdr[1])
                    threshs[obj] = thresh ## add object info
    
    else:
        pass

    return threshs