import numpy as np
from exposure.exposure import community_expo
from detectors.activator import activator
from clusteror.clustering import train_clusteror
from regressor.features import build_features
from regressor.regression import train_regressor

def situ_trainer(situ_name, train_set, gt_situ_expos, vis_concepts, clusteror, regressor, cfg):
    """
    Train an visual privacy exposure predictor on a situation

    :param situ_name:
    :param train_set:
    :param gt_situ_expos:
        train user exposure
        in a given situation
    :param clusteror:
    :param regressor:
    :param cfg
    :param verbose:
    :return:
         trained cluster model for the situation
         trained regression model for the situation
    """
    # Construct active detectors
    detectors, opt_threds = activator(vis_concepts, situ_name,\
                                      cfg.DATASETS.PRE_VIS_CONCEPTS, cfg.DETECTOR.LOAD)
    # Photo exposures of users
    commu_expo_features = community_expo(train_set, cfg.SOLVER.F_TOP,\
                                       detectors, opt_threds, cfg.DETECTOR.LOAD, cfg.SOLVER.FILTERING)
    # Build exposure features for users
    # by clustering their photo exposures
    trained_clusteror = train_clusteror(clusteror, commu_expo_features, cfg)
    # Build regression features for users
    reg_train_features, gt_train_expos = build_features(trained_clusteror, commu_expo_features, gt_situ_expos, cfg)
    # Fit to the regressor
    trained_regressor = train_regressor(regressor, reg_train_features, gt_train_expos, cfg)

    return detectors, opt_threds, \
           trained_clusteror, trained_regressor