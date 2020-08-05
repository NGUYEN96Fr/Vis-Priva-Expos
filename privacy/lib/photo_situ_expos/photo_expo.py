def photo_expo(photo, f_top, detectors, opt_threshs):

    """Estimate photo exposure
    
    Parameters
    ----------
    photo : dict
        objects in photo and its detection score
            {class1: [obj1, obj2,...], ... }

    f_top : float [0,1)
        A top N ranked detection object scores

    detectors : dict
        active ditector in a given situation and its score
            {detector1: score, ...}

    opt_threshs:
            optimal thresholds for each object. Precomputed by the base line privacy method.
    Returns
    -------
        expo_obj : tuple
            photo exposure and its objectness sum
                {exp +, expo -, objness}
    """
    expo_pos = 0 #positive exposure
    expo_neg = 0 #negative exposure

    sum_objectness = 0
    sum_pos_objectness = 0
    sum_neg_objectness = 0

    # for object_, scores in photo.items():
    #     if object_ in detectors:
    #
    #         objectness = sum([score for score in scores if score >= f_top])
    #         sum_objectness += objectness
    #
    #         if detectors[object_] >= 0:
    #             expo_pos += objectness*detectors[object_]
    #         else:
    #             expo_neg += objectness*detectors[object_]
    #
    # expo_obj = (0 , 0, 0) # no interesting objects

    for object_, scores in photo.items():
        if object_ in detectors:

            objectness = sum([score for score in scores if score >= opt_threshs[object_]])
            sum_objectness += objectness

            if detectors[object_] >= 0:
                expo_pos += objectness * detectors[object_]
                sum_pos_objectness += objectness
            else:
                expo_neg += objectness * detectors[object_]
                sum_neg_objectness += objectness

    expo_obj = (0, 0, 0)  # no interesting objects

    if sum_neg_objectness != 0: # if have
        expo_neg = expo_neg / sum_neg_objectness

    if sum_pos_objectness != 0:
        expo_pos = expo_pos / sum_pos_objectness

    expo_obj = (expo_pos, expo_neg, sum_objectness)

    return expo_obj