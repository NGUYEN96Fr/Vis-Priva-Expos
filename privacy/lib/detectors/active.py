import os
import json

def active_detectors(obj_expo_situ, situ_name, load_active_detectors):
    """Discover active detectors per situation

    Parameters
    ----------
    obj_expo_situ : dict
        object exposure scores in a given situation
            {class1: score, ...}

    load_active_detectors : dict
        load active detectors determined in baseline method

    Returns
    -------
        active_detector_situ : dict
            active detectors in a given situation, and its exposure scores
                {detector1: score1,...}
    """
    active_detector_situ = {}

    if not load_active_detectors:
        for class_, score in obj_expo_situ.items():
            if abs(score) >= 0.01:
                active_detector_situ[class_] = score

    else:
        root = os.path.dirname(os.path.dirname(os.getcwd()))
        optimal_thres_situs = json.load(open(os.path.join(root, 'privacy_baseline', 'out', 'optimal_thres_situs.txt')))
        detector_situ = optimal_thres_situs[situ_name]

        for object_, tau_thresh_score in detector_situ.items():
            active_detector_situ[object_] = tau_thresh_score[2]

    return active_detector_situ
