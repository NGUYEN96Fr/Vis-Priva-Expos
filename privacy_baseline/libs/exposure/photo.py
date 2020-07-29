def photo_exposure(photo, detectors, infer_mode):
    """

    :param photo: dict
        {class1: [obj1, ...], ...}

    :param detectors: dict
        {detector1: (threshold, object_score), ...} for not inference_mode
        {detector1: object_score, ...} for inference_mode

    :param infer_mode: boolean
        if in the inference mode, if not in searching an optimal subset of classes

    :return:
        activate : boolean
            does the photo have at least one detector

        photo_score: float
    """

    active_state = False
    photo_score = 0
    if not infer_mode:
        for class_, obj_scores in photo.items():
            if class_ in detectors:
                if max(obj_scores) >= detectors[class_][0]:
                    active_state = True
                    photo_score += max(obj_scores)*detectors[class_][1]

    else:
        for class_, obj_scores in photo.items():
            if class_ in detectors:
                active_state = True
                photo_score += max(obj_scores)*detectors[class_]

    return photo_score, active_state