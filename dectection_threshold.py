# -*- coding: utf-8 -*-
"""Docstring for the detection_threshold.py module

The module calculates objectness detection thresholds based on predefined false discovery positives (FDRs) for each category.

Note
----


Attributes
----------


"""

import os
import sys
import configparser
import json
import numpy as np
from kneed import KneeLocator
from utils import bb_intersection_over_union, prepro_files


def sel_prediction(imgs_gtbboxes,imgs_prdbboxes,iou_thresh):
    """Select predictions with an IoU > a threshold

    Parameters
    ----------
    imgs_gtbboxes : dict
        Imgs and its ground truth bounding boxes
    imgs_prdbboxes : dict
        Imgs and its predicted bounding boxes
    iou_thresh : float
        The threshold for the IoU
    
    Returns
    -------
    Sorted_Faoprd : dict
        False category predictions and  its sorted corresponding objectness scores
    Sorted_Toprd : dict
        True category predictions and  its sorted corresponding objectness scores

    """

    Faoprd = {} #false obj preds
    Sorted_Faoprd = {}
    Toprd = {} #true obj preds
    Sorted_Toprd = {} 
    
    for img, gt_objs in imgs_gtbboxes.items(): #get each img with its gt obj types
        
        if img in imgs_prdbboxes: #check if the img was predicted ?
            prd_objs = imgs_prdbboxes[img] #prd obj types of the img
            
            for gt_obj, gt_bboxes in gt_objs.items(): #get each obj type and its gt bboxes    
                
                for gt_bbox in gt_bboxes: #each gt bbox               

                    for prd_obj, prd_bboxes in prd_objs.items(): #consider all the prd bboxes

                        for prd_bbox_ in prd_bboxes:
                            objness = prd_bbox_[0]
                            prd_bbox = prd_bbox_[1:] 
                            iou = bb_intersection_over_union(gt_bbox,prd_bbox)
                            
                            if iou >= iou_thresh: #take only a satifisied iou
                                if gt_obj == prd_obj: #good prd !
                                    if gt_obj not in Toprd:
                                        Toprd[gt_obj] = []                                                                            
                                    Toprd[gt_obj].append(objness)
                                else:
                                    if gt_obj not in Faoprd:
                                        Faoprd[gt_obj] = []
                                    #if objness <= 0.5: # excluding condition, reference to (*) error
                                    Faoprd[gt_obj].append(objness)
    
    ## sort the objectness prediction scores
    for obj,scores in Faoprd.items():
        Sorted_Faoprd[obj] = sorted(scores) # ascending order
    
    for obj,scores in Toprd.items():
        Sorted_Toprd[obj] = sorted(scores)

    ##save to files
    with open('selected_false_predictions.json','w') as fp:
        json.dump(Sorted_Faoprd,fp)
    with open('selected_true_predictions.json','w') as fp:
        json.dump(Sorted_Toprd,fp)

    return Sorted_Faoprd,Sorted_Toprd


def objness_thresh(Faoprd,Toprd,FDRs,save_name,indi_thresh_flag):
    """Calculate objness thresholds for FDRs
    
    Parameters
    ----------
    Faoprd : dict
        False category predictions and  its sorted corresponding objectness scores
    Toprd : dict
        True category predictions and  its sorted corresponding objectness scores
    FDRs : list
        A list of false discovery rates 
    save_name: string
        saved file name
    indi_thresh_flag : bool
        Dont save the current result (if true), the indiviual threshold method is used

    Returns
    -------    
    dict
        Objects and its objness thresholds for the FDRs

    Notes
    -----
        Exclude all false predictions whose objectnesses are greater than 0.8 (two categories indicating the same physical object),
        to avoid 0/0, when the objectness threshold is equal to one. In this case, neither Faoprd nor Toprd instances having the objectness
        scores equal to 1 (*).
    """
    
    objness_threshs = {} #objectness thresholds
    objness_ranges = [0.01*i for i in range(30,101)] #objectness ranges
    #objness_ranges = [0.01*i for i in range(100,-1,-1)] #objectness ranges
    if not indi_thresh_flag:
        writer = open('./%s'%save_name,'w') #write to a .txt file
        writer.write('FDR ')

    for FDR in FDRs:
        
        if not indi_thresh_flag:
            if FDR != FDRs[-1]:
                writer.write('%s '%FDR)
            else:
                writer.write('%s\n'%FDR)

    for obj, Tscores in Toprd.items(): #object and true prediction scores
        if obj in Faoprd: 
            Fscores = Faoprd[obj] #false prediction scores
            if not indi_thresh_flag:
                writer.write('%s '%obj)
            objness_threshs[obj] = []
            #nb of predicted objs
            N = len(Fscores) + len(Tscores)

            for FDR in FDRs:

                for objness in objness_ranges:
                    #nb of false predictions, whose scores are greater than an objness threshold 
                    F = np.where(np.asarray(Fscores) >= objness)[0].shape[0] 
                    #nb of true predictions, whose scores are greater than an objness threshold 
                    T = np.where(np.asarray(Tscores) >= objness)[0].shape[0]
                    #calculate the corresponding FDR
                    #cFDR = F/(F+T+ 0.0001)
                    cFDR = F/N
                    if cFDR <= FDR:
                        break

                objness_threshs[obj].append(objness)

                if not indi_thresh_flag:
                    if FDR == FDRs[-1]:
                        writer.write('%s\n'%str(round(objness,2)))
                    else:
                        writer.write('%s '%str(round(objness,2)))
        else:
            sml_thresh = objness_ranges[0]
            objness_threshs[obj] = []
            if not indi_thresh_flag:
                writer.write('%s '%obj)
            
            for k in range(len(FDRs)):
                objness_threshs[obj].append(sml_thresh)
                
                if not indi_thresh_flag:
                    if k != len(FDRs) - 1:
                        writer.write('%s '%sml_thresh)
                    else:
                        writer.write('%s\n'%sml_thresh)
    
    for obj, Fscores in Faoprd.items():

        if obj not in Toprd:
            if not indi_thresh_flag:
                writer.write('%s '%obj)
            objness_threshs[obj] = []

            for FDR in FDRs:

                for objness in objness_ranges:
                    F = np.where(np.asarray(Fscores) >= objness)[0].shape[0] 
                    cFDR = F/len(Fscores)                   
                    if cFDR <= FDR:
                        break

                objness_threshs[obj].append(objness)
                
                if not indi_thresh_flag:
                    if FDR == FDRs[-1]:
                        writer.write('%s\n'%str(round(objness,2)))
                    else:
                        writer.write('%s '%str(round(objness,2)))             
                        
    if not indi_thresh_flag:
        writer.close()
    
    return objness_threshs


def opt_thresh_class(objness_threshs,FDRs):
    """Optimal objectness threshold per class
    
    Paramters
    ---------
    objness_threshs : dict
        Objects and its objness thresholds for the FDRs
    FDRs : list
        Predefined false discovery rates
    
    Returns
    -------
    dict
        Objects, its optimal objectness thresholds

    Notes
    -----
        Kneedle method used to select the optimal thresholds
    
    References
    ----------
        https://github.com/arvkevi/kneed

    """ 
    opt_threshs = {}
    writer = open('./individual_threshold.txt','w') #write to a .txt file
    writer.write('Class Thresh FDR\n')
        
    for obj, threshs in objness_threshs.items():
        kneedle = KneeLocator(FDRs, threshs, curve='convex', direction='decreasing', online= True, S = 1, interp_method='interp1d')
        if kneedle.elbow is not None:
            fdr = round(kneedle.elbow,2)
            if kneedle.knee_y is None:
                print('knee_y error: ',obj) ##can not file the elbow point for this category
                fdr = FDRs[-1]              ##take the threshold corresponding to the biggest FDR value 
                thresh = threshs[-1]        ##for that category
            else:    
                thresh = round(kneedle.knee_y,2)
        else:
            fdr = FDRs[0]
            thresh = threshs[0]
        writer.write('%s %s %s\n'%(obj,thresh,fdr))
    
    writer.close()

                
def main():
    """
    
    """
    ##read the param config file
    conf = configparser.ConfigParser() 
    conf.read(sys.argv[1])
    conf = conf[os.path.basename(__file__)]
    
    ##get params
    gt_path  = conf['gt_path'] 
    prd_path = conf['prd_path']
    load_bb = conf.getboolean('load_bb')
    iou_thresh = conf.getfloat('iou_thresh')
    load_sel_prd = conf.getboolean('load_sel_prd')
    indi_thresh_flag = conf.getboolean('indi_thresh_flag')
    save_name = conf['sav_name']

    if indi_thresh_flag == True:
        FDRs = [0.01*i for i in range(21)]  ## FDRs list
    else:
        FDRs_ = conf['FDRs'].split('_')
        FDRs = [float(fdr) for fdr in FDRs_]

    if not load_bb:
        print('Preprocess files !')
        imgs_gtbboxes = prepro_files(gt_path, is_gt= True) 
        imgs_prdbboxes = prepro_files(prd_path, is_gt= False) 
    else:
        print('Load preprocessed files !')
        with open('preprocessed_'+gt_path.split('/')[-1].split('.')[0]+'.json','r') as fp:
            imgs_gtbboxes = json.load(fp)
        with open('preprocessed_'+prd_path.split('/')[-1].split('.')[0]+'.json','r') as fp:
            imgs_prdbboxes = json.load(fp)

    if not load_sel_prd:
        print('Select predictions !')
        Faoprd, Toprd = sel_prediction(imgs_gtbboxes,imgs_prdbboxes,iou_thresh)
    else:
        print('Load selected predictions !')
        with open('selected_false_predictions.json','r') as fp:
            Faoprd = json.load(fp)
        with open('selected_true_predictions.json','r') as fp:
            Toprd = json.load(fp)
    
    ##objness score thresholds
    print('Thresholds - FDRs !')
    objness_threshs = objness_thresh(Faoprd,Toprd,FDRs,save_name,indi_thresh_flag)
    ## optimal threshold per a class
    if indi_thresh_flag:
        print('Optimal Threshold per a class !')
        opt_thresh_class(objness_threshs,FDRs)

if __name__ == "__main__":
    main()