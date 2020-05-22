# -*- coding: utf-8 -*-
"""Docstring for the user_exposure.py module

The module estimates user visual privacy exporsure in a community

Note
----


Attributes
----------

"""

import os
import json
import configparser
import numpy as np
from utils import *

def user_vis_prav_expos(users_meta, situ_scores, threshs):
    """Estimate user visual privacy exposure in a situation

    Parameters
    ----------
    users_meta : dict
        users and its meta data
            {'user_key1': { 'img_id1': { 'category1': [ ( objness1 , [bbx1] ), ... ], ... }, ... }, ... }
    situ_scores : dict
            {'object 1' : (score, domain, is_threat), ...}
    threshs : dict
        objectness thresolds
            {'object 1': float, ... }
    
    Returns
    -------
    user_expos : dict
        user exposure
            {'user_key 1': {'total_expos' : float, 'domain_expos' : { 'domain 1' : float, ... }}, ... }

    Notes
    -----
        For each category in an image, choose the box with the maximum objectness.


    """
    user_expos = {}
    cate_not_in_situ = ()
    cat_not_in_obj_thresh = ()

    for user_key, user_imgs in users_meta.items(): #each user
        user_expos[user_key] = {}
        user_expos[user_key]['total_expos'] = 0.0 #total exposure
        user_expos[user_key]['domain_expos'] = {} #exposure per a domain 

        for img_id, img_meta in user_imgs.items(): #each user img
            
            for cate, bboxes in img_meta.items(): #each img category
                
                if cate in situ_scores: #check if the category in the considered situation
                    max_objness = -1 #max img cate objectness
                    ref_box = [] #corresponding box
                    
                    for box_ in bboxes: #select the max objness cate box ??
                        objness = box_[0]
                        box = box_[1]
                        
                        if objness > max_objness:
                            max_objness = objness
                            ref_box = box

                    if cate in threshs:
                        cate_thresh =  threshs[cate] #objness threshold for the category
                        
                        if max_objness >= cate_thresh:
                            obj_domain = situ_scores[cate][1]
                            situ_obj_score = situ_scores[cate][0]# object score per a situ
                            expos = max_objness*situ_obj_score # *** user img object exposure
                            
                            if obj_domain not in  user_expos[user_key]['domain_expos']: #check domain exist, update user exposure
                                user_expos[user_key]['domain_expos'][obj_domain] = 0.0
                            
                            user_expos[user_key]['domain_expos'][obj_domain] += expos
                            user_expos[user_key]['total_expos'] += expos
                    
                    else:
                        #cat_not_in_obj_thresh.append(cate)
                        pass
                
                else:
                    #cate_not_in_situ.append(cate)
                    pass

    return user_expos


def ranking_user(user_expos, situ, export_path):
    """Ranking a user among of comunity's users in a situation, or including all situations. 

    Parameters
    ----------
    user_expos : dict
        user visual privacy exposure
            {'user_key 1': {'total_expos' : float, 'domain_expos' : {'domain 1' : float, ... }}, ... }
    situ : string
        situation
    export_path : string
        export path
    
    Returns
    -------
    dict
        user visual privacy and its ranking
            {'user_key 1': (total_expos, ranking), ...}


    """
    writer = open('%s%s_user_expos_ranking.txt'%(export_path,situ),'w')
    writer.write('user_id         expos    rank\n') #4 blanks

    user_total_expos = []
    userkeys = []
    user_rank = {}

    for userkey, expos in user_expos.items():
        userkeys.append(userkey)
        user_total_expos.append(expos['total_expos'])

    ##desceding exposure scores
    des_score_indexes = np.argsort(user_total_expos)[::-1] 
    nb_users = des_score_indexes.shape[0] #nb of users
    
    ranking = 1 #cur user ranking
    for index in list(des_score_indexes):
        user_rank[userkeys[index]] = (user_total_expos[index],ranking)
        writer.write('%s    %s    %s/%s\n'%(userkeys[index],round(user_total_expos[index],2),ranking,nb_users)) #4 blanks
        ranking += 1
    
    return user_rank


def user_expo_situations(situs,export_path):
    """Ranking user visual privacy expos in all situations

    Parameters
    ----------
    situs : list of dict
        List of user expos in each situation
            [situ1, ...]
            situ1 = {'user_key 1': {'total_expos' : float, 'domain_expos' : {'domain 1' : float, ... }}, ... }
    export_path : string
        export path

    Returns
    -------
    dict
        User ranking in all situations
            {'user_key 1': (total_expos, ranking), ...}

    """
    
    situ_ = 'all_situ'
    user_expos = {}

    ##sum all situation scores
    for situ in situs:
    
        for user_key, expos in situ.items():
            
            if user_key not in user_expos:
                user_expos[user_key] = { }
                user_expos[user_key]['total_expos'] = 0.0
                user_expos[user_key]['domain_expos'] = {}

            user_expos[user_key]['total_expos'] += expos['total_expos']

            for domain, score in expos['domain_expos'].items():
                
                if domain not in user_expos[user_key]['domain_expos']:
                    user_expos[user_key]['domain_expos'][domain] = 0.0
                
                user_expos[user_key]['domain_expos'][domain] += score
    

    ##ranking users in all situations
    user_rank = ranking_user(user_expos,situ_,export_path)

    return user_rank


def main():
    """
    
    """
    
    ##read the param config file
    conf = configparser.ConfigParser() 
    conf.read(sys.argv[1])
    conf = conf[os.path.basename(__file__)]

    ##get params
    situ_path = conf['situ_path']
    user_path = conf['user_path']
    obj_path = conf['obj_path']
    method = conf['method'] #objectness threshold method
    sav_path = conf['sav_path']

    if not os.path.exists(sav_path):
        os.makedirs(sav_path)

    ##extract user meta
    print('Preprocess user data file !')
    users_meta = preproc_user_file(user_path)
    print('Preprocess objectness threshold file !')
    threshs = prepro_threshs(obj_path, sav_path, method)

    ##extract situations
    print("Preprocess situation files !")
    situ_files = os.listdir(situ_path)

    user_expo_situs = []
    export_path = sav_path + method +'/'
    for situ_file in situ_files: #read each situ file
        file_path = situ_path + situ_file
        situ_name, situ_scores = prepro_obj_score(file_path) #situation name, and its situ scores
        #estimating user visual privacy socres
        print('Situation: ',situ_name)
        user_expos = user_vis_prav_expos(users_meta, situ_scores, threshs)
        #ranking user
        _ = ranking_user(user_expos, situ_name, export_path)
        user_expo_situs.append(user_expos)
    
    ##evaluate user visual privacy exposure in all situations
    _ = user_expo_situations(user_expo_situs,export_path)
        

if __name__ == "__main__":
    main()
