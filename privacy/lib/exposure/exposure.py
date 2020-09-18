def photo_expo(photo, f_top, detectors, opt_threshs, load_detectors):
    """Estimate photo exposure

    Parameters
    ----------
    photo : dict
        objects in photo associating its detection confidence
            {class1: [obj1, obj2,...], ... }

    f_top : float [0,1)
        A top N ranked detection object confidence

    load_detectors : boolean
        load active detectors pre-computed by the privacy
        base-line method

    detectors : dict
        active detectors in a given situation and its score
            {detector1: score, ...}


    opt_threds:
            optimal threshold for each object. Precomputed by the base line privacy method.
    Returns
    -------
        expo_obj : tuple
            photo exposure and its objectness sum
                {exp +, expo -, objness}
    """
    expo_pos = 0  # positive exposure
    expo_neg = 0  # negative exposure

    sum_objectness = 0
    sum_pos_objectness = 0
    sum_neg_objectness = 0

    for object_, scores in photo.items():
        if object_ in detectors:

            if not load_detectors:
                objectness = sum([score for score in scores if score >= f_top])
            else:
                objectness = sum([score for score in scores if score >= opt_threshs[object_]])

            sum_objectness += objectness

            if detectors[object_] >= 0:
                expo_pos += objectness * detectors[object_]
                sum_pos_objectness += objectness
            else:
                expo_neg += objectness * detectors[object_]
                sum_neg_objectness += objectness

    if sum_neg_objectness != 0:  # if have
        expo_neg = expo_neg / sum_neg_objectness

    if sum_pos_objectness != 0:
        expo_pos = expo_pos / sum_pos_objectness

    expo_obj = (expo_pos, expo_neg, sum_objectness)

    return expo_obj


def user_expo(user_photos, f_top, detectors, opt_threshs, load_detectors, filter):
    """Estimate user exposure

    Parameters
    ----------
        user_photos : dict
            user photos associating with predicted object confidence
                {photo1: {class1: [obj1, ...], ...},...}

        load_detectors : boolean
            load active detectors pre-computed by the privacy
            base-line method

        f_top : float [0,1)
            A top N ranked detected object confidence

        detectors : dict
            active detectors for a given situation

        filter : boolean
            filtering neutral photos with a threshold 0.01
        
    Returns
    -------
        expo : dict
            user exposure
                {photo1: ( expo +, expo -, photo_objectness),...}

    """
    expo = {}

    for photo in user_photos:
        pos_expo, neg_expo, objectness =  photo_expo(user_photos[photo], f_top, detectors, opt_threshs, load_detectors)
        if filter:
            if abs(pos_expo) + abs(neg_expo) >= 0.01:
                expo[photo] = photo_expo(user_photos[photo], f_top, detectors, opt_threshs, load_detectors)

        else:
            expo[photo] = photo_expo(user_photos[photo], f_top, detectors, opt_threshs, load_detectors)

    return expo


def community_expo(users, f_top, detectors, opt_threshs, load_detectors, filter = False):
    """Estimate photo exposure for all users in a given situation

    Parameters
    ----------
        users : dict
            users and their photos
                {user1: {photo1: {class1: [obj1, ...], ...}, ...}, ...}
        
        f_top : float [0,1)
            A top N ranked object detection confidence

        load_detectors : boolean
            load active detectors pre-computed by the privacy
            base-line method

        detectors : dict
            active detectors for a given situation

        opt_threshs : dict
            optimal active detector thresholds
                {detector1: thresh1, ...}

        filter : boolean
            if filtering neutral images

    Returns
    -------
        expo : dict
            community exposure
            {user1: {photo1: (expo +, expo-, sum_objectness), ...}, ...}

    """
    expo = {}

    for user, photos in users.items():
        expo[user] = user_expo(photos, f_top, detectors, opt_threshs, load_detectors, filter)

    return expo