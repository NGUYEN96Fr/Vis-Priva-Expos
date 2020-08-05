import math
import tqdm
import numpy as np
from optimal_search.correlation import corr, pos_neg_corr

def search_thres(train_data, gt_user_expo, detector_score, corr_type):
    """

    :param train_data:
    :param gt_user_expo:
    :param detector_score: list
            [detector, object_score]
                + detector: the type of object need to searched for
                + object_score: crowd-sourcing object score
    :param corr_type:

    :return:
        the best threshold for the given object

    """
    threshold_list = [float("{:.2f}".format(0.01*i)) for i in range(101)]
    tau_list = []

    for threshold in threshold_list:
        detector = {detector_score[0]: (threshold, detector_score[1])}
        tau = corr(train_data, gt_user_expo, detector, corr_type)
        # tau = pos_neg_corr(train_data, gt_user_expo, detector, corr_type)
        if math.isnan(tau):
            tau = 0
        tau_list.append(tau)

    tau_max = max(tau_list)
    threshold_max = threshold_list[np.argmax(tau_list)]

    return tau_max, threshold_max


def search_optimal_thres(train_data, gt_user_expo, detectors, corr_type):
    """Search optimal object thresholds
    for all type of object within a given correlation type 
    
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
    max_tau_detectors = {}

    for detector, score in tqdm.tqdm(detectors.items()):
        detector_score = [detector,score]
        tau_max, threshold_max = search_thres(train_data, gt_user_expo, detector_score, corr_type)
        max_tau_detectors[detector] = (tau_max, threshold_max, score)

    return max_tau_detectors



def export_tau_ranking(opt_thresh_situ, save_file):
    """

    :param opt_thresh_situ:
        {situ1:  {object1: (tau_max_1, threshold1, score1), ...}, ...}

    :return:

    """
    for situ, objects in opt_thresh_situ.items():
        objects_ = []
        taus = []
        threshs = []
        scores = []

        for object, tau_thresh_score in objects.items():

            objects_.append(object)
            taus.append(tau_thresh_score[0])
            threshs.append(tau_thresh_score[1])
            scores.append(tau_thresh_score[2])
        taus2 = [float(format(x,'.2g')) for x in taus]
        writer = open('%stau_obj_ranking_%s'%(save_file,situ),'w')
        writer.write('object\ttau\tthreshold\tscore\tranking\n')

        sorted_indexes = list(np.argsort(np.asarray(taus2))[::-1])
        N = len(sorted_indexes)

        for i, index_ in enumerate(sorted_indexes):
            if abs(scores[index_]) >= 1:
                writer.write('%s\t%s\t%s\t%s\t%s/%s\n'%(objects_[index_], taus2[index_], threshs[index_], scores[index_], i, N))

        writer.close()