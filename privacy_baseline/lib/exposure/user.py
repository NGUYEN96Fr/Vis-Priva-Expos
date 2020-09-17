from exposure.photo import photo_exposure, photo_pos_neg_expo

def user_expo(photos, detectors, test_mode):
    """Estimate user exposure

    :param photos: dict
        user photos and its detected objects
            {photo1: {class1: [obj1, ...], ...}}, ...}

    :param detectors: dict
         {detector: (threshold, object_score),...} for not inference_mode
        {detector1: object_score, ...} for inference_mode

    :return:
        user_score: float
    """

    user_score = 0
    cardinality = 0
    for photo, detected_objects in photos.items():

        photo_score, active_state = photo_exposure(detected_objects, detectors, test_mode)
        user_score += photo_score

        if active_state:
            cardinality += 1

    if cardinality != 0:
        user_score = user_score/cardinality

    return user_score

def user_expo_situ(users, detectors, test_mode):
    """

    :param users:
        users in a situation and its photos
            {user1: {photo1: {class1: [obj1, ...], ...}}, ...}, ...}

    :param detectors: dict
        {detector1: (threshold, object_score), ...}

    :return:
        community_expo: dict
            {user1: score, ...}

    """
    community_expo = {}
    for user, photos in users.items():
        community_expo[user] = user_expo(photos, detectors, test_mode)

    return community_expo


def user_pos_neg_expo(photos, detectors, test_mode):
    """

    :param photos:
    :param detectors:
    :return:
    """
    pos_user_score = 0
    neg_user_score = 0
    cardinality_pos = 0
    cardinality_neg = 0

    for photo, detected_objects in photos.items():

        pos, neg , active_pos, active_neg = photo_pos_neg_expo(detected_objects, detectors, test_mode)
        pos_user_score += pos
        neg_user_score += neg

        if active_pos:
            cardinality_pos += 1

        if active_neg:
            cardinality_neg += 1

    if cardinality_pos != 0:
        pos_user_score = pos_user_score/cardinality_pos

    if cardinality_neg != 0:
        neg_user_score = neg_user_score/cardinality_neg

    return [pos_user_score, neg_user_score]


def pos_neg_user_expo_situ(users, detectors, test_mode):
    """

    :param users:
    :param detectors:
    :return:
    :return:
        community_expo: dict
            {user1: [pos, neg], ...}
    """

    community_expo = {}
    for user, photos in users.items():
        community_expo[user] = user_pos_neg_expo(photos, detectors, test_mode)

    return community_expo