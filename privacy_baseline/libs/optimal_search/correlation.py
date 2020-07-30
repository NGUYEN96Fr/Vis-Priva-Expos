import numpy as np
from scipy.stats import kendalltau, pearsonr
from exposure.user import user_expo_situ

def corr(train_data, gt_user_expo, detectors, corr_type, infer_mode = False, print_ = False):
    """Calculate correlation score for a threshold

    :param train_data: dict
        users and images in training data
            {user1: {photo1: {class1: [obj1, ...], ...}}, ...}, ...}

    :param gt_user_expo: dict
        user expo in a given situation
            {user1: avg_score, ...}

    :param detectors: dict
        the type of object need to searched for
            {detector: (thres, object_score), ...} for not inference_mode
                + thres: a given considered threshold
                + object_score: crowd-sourcing object score
            {detector1: object_score, ...} for inference_mode

    :param corr_type: string
        correlation type:
            + pear_corr
            + kendall_corr

    :param infer_mode: boolean
        if in the inference mode, if not in searching an optimal subset of classes


    :return:
        tau: float
            correlation
    """
    user_scores = user_expo_situ(train_data, detectors, infer_mode)
    automatic_eval = []
    manual_eval = []

    for user, score in user_scores.items():
        automatic_eval.append(score)
        manual_eval.append(gt_user_expo[user])

    automatic_eval = np.asarray(automatic_eval)
    manual_eval = np.asarray(manual_eval)

    if print_:
        print('Automatic Eval: ')
        print(automatic_eval)
        print('Manual Eval: ')
        print(manual_eval)

    if corr_type == 'pear_corr':
        tau, _ = pearsonr(automatic_eval,manual_eval)
    elif corr_type == 'kendall_corr':
        tau, _ = kendalltau(automatic_eval, manual_eval)

    return tau
