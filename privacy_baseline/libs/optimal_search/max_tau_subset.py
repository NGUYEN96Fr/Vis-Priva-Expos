import math
import tqdm
import numpy as np
from optimal_search.correlation import corr


def select_subset(detectors, tau_fix):
    """
    Select a subset whose detector taus are greater than tau_fix

    :param: detectors: dict
        {detector1: (tau_max1, threshold1, score1), ...}

    :param: tau_fix: float

    :return:
        tau_detectors: dict
            {detector1: (threshold1, score1), ...}

    """
    supp_info = {}
    detector_subset = {}

    for detector, tau_thres_score in detectors.items():
        if tau_thres_score[0] >= tau_fix:
            detector_subset[detector] = (tau_thres_score[1], tau_thres_score[2])
            supp_info[detector] = (tau_thres_score[1], tau_thres_score[2])
    
    return detector_subset, supp_info

def cross_validation(users, gt_user_expo, detector_subset, corr_type, k_fold = 5):
    """

    :param users:
    :param gt_user_expo:
    :param detector_subset:
    :param corr_type:
    :param k_fold: number of folds
    :return:
    """
    test_fold_size = int(len(list(users.keys()))/k_fold)

    for index in range(k_fold):
        start = index*test_fold_size
        end = (index + 1)*test_fold_size
        count = 0
        train_fold = {}
        test_fold = {}
        for user, photos in users.items():
            if count >= start and count < end:
                test_fold[user] = photos
            else:
                train_fold[user] = photos


def tau_subset(users, gt_user_expo, detectors, corr_type):
    """
    Estimate the best correlation score for a subset tau_detectors

    :param: users
        users in a situation and its photos
            {user1: {photo1: {class1: [obj1, ...], ...}}, ...}, ...}
            
    :param gt_user_expo: dict
        user expo in a given situation
            {user1: avg_score, ...}
    
    :param: detectors: dict
        {detector1: (tau_max1, threshold1, score1), ...}
    
    :param corr_type: string

    :return:

    """
    opt_detectors = []
    tau_estimate_list = []
    tau_fixes = list(np.linspace(-1,1,201))
    
    for tau_fix in tqdm.tqdm(tau_fixes):
        detector_subset, sup_info = select_subset(detectors, tau_fix) #select subset
        ## TODO, apply cross validation
        tau_est = corr(users, gt_user_expo, detector_subset, corr_type)
        if math.isnan(tau_est):
            tau_est = -1
        tau_estimate_list.append(tau_est)
        opt_detectors.append(sup_info)

    tau_max = max(tau_estimate_list)
    supp_info = opt_detectors[np.argmax(tau_estimate_list)]
    threshold = tau_fixes[np.argmax(tau_estimate_list)]

    return tau_max , supp_info, tau_estimate_list, threshold