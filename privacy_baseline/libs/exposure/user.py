from photo import photo_exposure

def user_expo(photos, detectors, infer_mode, t_max_O, t_max_D):
    """Estimate user exposure

    :param photos: dict
        user photos and its detected objects
            {photo1: {class1: [obj1, ...], ...}}, ...}

    :param detectors: dict
         {detector: (threshold, object_score)} for not inference_mode
        {detector1: object_score, ...} for inference_mode

    :param infer_mode: boolean
        if in the inference mode, if not in searching an optimal subset of classes

    :param t_max_O: dict
        object and its best correlation score (individually taking)

    :param t_max_D: float
        the best correlation score of an optimal subset of objects in a given situation

    :return:
        user_score: float
    """

    user_score = 0
    carinality = 0
    for photo, detected_objects in photos.items():

        photo_score, active_state = photo_exposure(detected_objects, detectors, infer_mode, t_max_O, t_max_D)
        user_score += photo_score

        if active_state:
            carinality += 1

    if carinality != 0:
        user_score = user_score/carinality

    return user_score

def user_expo_situ(users, detectors,  infer_mode = False, t_max_O = {}, t_max_D = 0):
    """

    :param users:
        users in a situation and its photos
            {user1: {photo1: {class1: [obj1, ...], ...}}, ...}, ...}

    :param detectors: dict
         {detector: (threshold, object_score)} for not inference_mode
        {detector1: object_score, ...} for inference_mode

    :param infer_mode: boolean
        if in the inference mode, if not in searching an optimal subset of classes

    :param t_max_O: dict
        object and its best correlation score (individually taking)

    :param t_max_D: float
        the best correlation score of an optimal subset of objects in a given situation

    :return:
        community_expo: dict
            {user1: score, ...}

    """
    community_expo = {}
    for user, photos in users.items():
        community_expo[user] = user_expo(photos, detectors, infer_mode, t_max_O, t_max_D)

    return community_expo