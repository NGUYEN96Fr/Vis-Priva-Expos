def photo_exposure(photo, detectors, f_top = 0.6):
    """

    :param photo: dict
        {class1: [obj1, ...], ...}

    :param detectors: dict
        {detector1: (threshold, object_score), ...}

    :return:
        activate : boolean
            does the photo have at least one detector

        photo_score: float
    """

    active_state = False

    photo_score = 0
    for class_, obj_scores in photo.items():
        if class_ in detectors:
            if max(obj_scores) >= detectors[class_][0]:
                active_state = True
                photo_score += max(obj_scores)*detectors[class_][1]

    # for class_, obj_scores in photo.items():
    #     if class_ in detectors:
    #         for score in obj_scores:
    #             if score >= detectors[class_][0] and score > f_top:
    #                 active_state = True
    #                 photo_score += score*detectors[class_][1]

    return photo_score, active_state