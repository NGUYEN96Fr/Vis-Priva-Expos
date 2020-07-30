def active_subset(detectors, tau_D_max):
    """
    Select a subset whose detector taus are greater than tau_fix

    :param: detectors: dict
        {detector1: (tau_max1, threshold1, score1), ...}

    :param: tau_D_max: float

    :return:
        tau_detectors: dict
            {detector1: score1, ...}

    """
    active_detectors = {}
    for detector, tau_thres_score in detectors.items():
        if tau_thres_score[0] >= tau_D_max:
            active_detectors[detector] =  tau_thres_score[2]

    print('Activated Detectors: ',active_detectors)
    
    return active_detectors