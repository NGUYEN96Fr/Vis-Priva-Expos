import numpy as np
from scipy.stats import kendalltau, pearsonr
from exposure.user import user_expo_situ


def corr(train_data, gt_user_expo, detector, corr_type):
    """Calculate correlation score for a threshold

    :param train_data: dict
        users and images in training data
            {user1: {photo1: {class1: [obj1, ...], ...}}, ...}, ...}

    :param gt_user_expo: dict
        user expo in a given situation
            {user1: avg_score, ...}

    :param detector: dict
        the type of object need to searched for
            {detector: (thres, object_score)}
                + thres: a given considered threshold
                + object_score: crowd-sourcing object score

    :param corr_type: string
        correlation type:
            + pear_corr
            + kendall_corr

    :return:
        tau: float
            correlation
    """
    user_scores = user_expo_situ(train_data, detector, infer_mode=False)
    automatic_eval = []
    manual_eval = []

    for user, score in gt_user_expo.items():
        automatic_eval.append(score)
        manual_eval.extend(user_scores[score])

    automatic_eval = np.asarray(automatic_eval)
    manual_eval = np.asarray(manual_eval)

    if corr_type == 'pear_corr':
        tau, _ = pearsonr(automatic_eval,manual_eval)
    elif corr_type == 'kendall_corr':
        tau, _ = kendalltau(automatic_eval, manual_eval)

    return tau



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
        tau_list.append(tau)

    tau_max = max(tau_list)
    threshold_max = threshold_list[np.argmax(tau_list)]

    return tau_max, threshold_max