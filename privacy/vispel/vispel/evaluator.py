from detectors.activator import activator

def situ_evaluator(clusteror, regressor, test_set, gt_situ_expo):
    """
    Evaluate the exposure predictor on a given situation

    Parameters
    ----------
    clusteror: trained clusteror
    regressor: trained regressor
    test_set:
    gt_situ_expo: user exposures in the given situation

    Returns
    -------

    """
    # Construct active detectors
    detectors, opt_threds = activator(vis_concepts, situ_name,\
                                      cfg.DATASETS.PRE_VIS_CONCEPTS, cfg.DETECTOR.LOAD)
