import math
import numpy as np
from optimal_search.correlation import corr


def search_thres(train_data, gt_user_expo, detectors, corr_type, cfg):
    """

    :param train_data:
    :param gt_user_expo:
    :param corr_type:
    :param detectors:
    :param cfg:
    :return:
        the best threshold for the given object

    """
    threshold_list = [float("{:.2f}".format(0.01 * i)) for i in range(101)]
    tau_list = []

    for threshold in threshold_list:
        tdetectors = {detector_score[0]: (threshold, detector_score[1]) for detector_score in detectors}
        tau = corr(train_data, gt_user_expo, tdetectors, corr_type, cfg)
        if math.isnan(tau):
            tau = 0
        tau_list.append(tau)

    tau_max = max(tau_list)
    opt_threshold = threshold_list[np.argmax(tau_list)]

    return tau_max, opt_threshold


def search_optimal_thres(train_data, gt_user_expo, detectors, corr_type, cfg):
    """Search optimal threshold for all detectors
    
    :param train_data: dict
        users and images in training data
            {user1: {photo1: {class1: [obj1, ...], ...}}, ...}, ...}

    :param gt_user_expo: dict
        user expo in a given situation
            {user1: avg_score, ...}

    :param detectors: dict
        all detectors in a given situation
            {detector1: score1, detector2: score2, ...}

    :param corr_type: string
        correlation type:
            + pear_corr
            + kendall_corr
    
    :return
        max_tau_detectors: dict
            {object1: (tau_max_1, threshold1, score1), ...}
    """
    list_detectors = []

    for detector, score in detectors.items():
        list_detectors.append([detector, score])

    tau_max, opt_thresh = search_thres(train_data, gt_user_expo, list_detectors, corr_type, cfg)

    return opt_thresh
