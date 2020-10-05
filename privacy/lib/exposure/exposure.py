from exposure.focal_exposure import focal_exposure as FE


def feature_transform(f_expo_pos, f_expo_neg, f_dens, transform):
    """
    Apply feature transform on photo features scaled by focal exposure

    :param f_expo_pos:
    :param f_expo_neg:
    :param f_dens:
    :param transform: transforming method
    :return:
        transformed features

    """
    if transform == 'ORG':

        return [f_expo_pos, f_expo_neg, f_dens]


def photo_expo(photo, f_top, detectors, opt_threshs, load_detectors, cfg):
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

    expo_pos = []  # positive exposure
    expo_neg = []  # negative exposure

    objectness = []

    attract_pos_concepts = []
    attract_neg_concepts = []
    neutral_pos_concepts = []
    neutral_neg_concepts = []

    for object_, scores in photo.items():
        if object_ in detectors:
            if detectors[object_] > 0.4:
                attract_pos_concepts.append(object_)
            elif detectors[object_] < -1:
                attract_neg_concepts.append(object_)

            if 0 <= detectors[object_] <= 0.4:
                neutral_pos_concepts.append(object_)
            if -1 <= detectors[object_] < 0:
                neutral_neg_concepts.append(object_)

    if len(neutral_pos_concepts) != 0:
        ratio = len(attract_pos_concepts)/len(neutral_pos_concepts)
        if ratio > 0 and ratio < 1/3:
            scale_pos_flag = True
        else:
            scale_pos_flag = False
    else:
        scale_pos_flag = False

    if len(neutral_neg_concepts) != 0:
        ratio = len(attract_neg_concepts)/len(neutral_neg_concepts)
        if ratio > 0 and ratio < 1/3:
            scale_neg_flag = True
        else:
            scale_neg_flag = False
    else:
        scale_neg_flag = False


    for object_, scores in photo.items():
        obj_score = 0
        if object_ in detectors:
            if not load_detectors:
                valid_obj = [score for score in scores if score >= f_top]
                if sum(valid_obj) > 0:
                    obj_score += sum(valid_obj)/len(valid_obj)
            else:
                valid_obj = [score for score in scores if score >= opt_threshs[object_]]
                if sum(valid_obj) > 0:
                    obj_score += sum(valid_obj) / len(valid_obj)
            objectness.append(obj_score)

            # Only scale object scores when object-ness is sufficiently high, and
            # exist numerous neutral objects as the same type (positive or negative)

            if detectors[object_] >= 0:
                #if scale_pos_flag and obj_score > 0.7:
                if obj_score > 0.7:
                    scaled_expo = FE(detectors[object_], cfg.SOLVER.GAMMA, cfg.SOLVER.K)
                else:
                    scaled_expo = detectors[object_]
                expo_pos.append(scaled_expo)

            if detectors[object_] <= 0:
                # if scale_neg_flag and obj_score > 0.7:
                if obj_score > 0.7:
                    scaled_expo = FE(detectors[object_], cfg.SOLVER.GAMMA, cfg.SOLVER.K)
                else:
                    scaled_expo = detectors[object_]
                expo_neg.append(scaled_expo)

    if sum(objectness) != 0:
        objectness = sum(objectness)/len(objectness)
    else:
        objectness = 0

    if sum(expo_pos) != 0:
        expo_pos = sum(expo_pos)/len(expo_pos)
    else:
        expo_pos = 0

    if sum(expo_neg) != 0:
        expo_neg = sum(expo_neg)/len(expo_neg)
    else:
        expo_neg = 0
    expo_obj = (expo_pos, expo_neg, objectness)

    scale_flag = scale_pos_flag + scale_neg_flag
    return expo_obj, scale_flag


def user_expo(user_photos, f_top, detectors, opt_threshs, load_detectors, cfg):
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
                {photo1: [transformed features],...}

    """
    expo = {}
    count_rescaled_imgs = []
    for photo in user_photos:
        (pos_expo, neg_expo, objectness), scale_flag =  photo_expo(user_photos[photo], f_top, detectors, opt_threshs, load_detectors, cfg)
        f_expo_pos = pos_expo
        f_expo_neg = neg_expo
        f_dens = objectness

        if scale_flag:
            count_rescaled_imgs.append(photo)

        if cfg.SOLVER.FILTERING:
            if abs(f_expo_pos) + abs(f_expo_neg) >= cfg.SOLVER.FILT:
                # Apply feature transform
                expo[photo] = feature_transform(f_expo_pos, f_expo_neg, f_dens, cfg.SOLVER.FEATURE_TYPE)
        else:
            # Apply feature transform
            expo[photo] = feature_transform(f_expo_pos, f_expo_neg, f_dens, cfg.SOLVER.FEATURE_TYPE)

    return expo


def community_expo(users, f_top, detectors, opt_threshs, load_detectors, cfg):
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

    Returns
    -------
        expo : dict
            community exposure
            {user1: {photo1: [transform features], ...}, ...}

    """
    expo = {}

    for user, photos in users.items():
        expo[user] = user_expo(photos, f_top, detectors, opt_threshs, load_detectors, cfg)

    return expo