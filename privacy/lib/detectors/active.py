def active_detectors(obj_expo_situ):
    """Discover active detectors per situation

    Parameters
    ----------
    obj_expo_situ : dict
        object exposure scores in a given situation
            {class1: score, ...}

    
    Returns
    -------
        active_detector_situ : dict
            active detectors in a given situation, and its exposure scores
                {detector1: score1,...}
    """
    active_detector_situ = {}

    for class_, score in obj_expo_situ.items():
        if abs(score) >= 0.01:
            active_detector_situ[class_] = score

    return active_detector_situ
