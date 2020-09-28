import numpy as np
from sklearn.decomposition import PCA
from exposure.exposure import community_expo
from detectors.activator import activator
from clusteror.clustering import train_clusteror
from regressor.features import build_features
from regressor.regression import train_regressor


def situ_trainer(situ_name, X_train_set, X_community, gt_situ_expos, vis_concepts, clusteror, regressor, cfg):
    """
    Train an visual privacy exposure predictor on a situation

    :param situ_name:
    :param X_train_set:
        user ids in the train set
    :param X_community:
        all user ids in the community
    :param gt_situ_expos:
        all user exposures
        in a given situation
    :param clusteror:
    :param regressor:
    :param cfg:
    :param verbose:


    """
    # Construct active detectors
    detectors, opt_threds = activator(vis_concepts, situ_name,\
                                      cfg.DATASETS.PRE_VIS_CONCEPTS, cfg.DETECTOR.LOAD)

    # Calculate photo exposures
    # for all user ids in the community
    commu_expo_features = community_expo(X_community, cfg.SOLVER.F_TOP,\
                                       detectors, opt_threds, cfg.DETECTOR.LOAD, cfg, cfg.SOLVER.FILTERING)

    # Calculate photo exposuers
    # for user ids in the train set
    train_expo_features = community_expo(X_train_set, cfg.SOLVER.F_TOP,\
                                       detectors, opt_threds, cfg.DETECTOR.LOAD, cfg, cfg.SOLVER.FILTERING)

    # Build exposure features for  all user ids
    # by clustering their photo exposures
    trained_clusteror = train_clusteror(situ_name, clusteror, commu_expo_features, cfg)

    # Build regression features for user ids in the trained set
    reg_train_features, gt_train_expos = build_features(trained_clusteror, train_expo_features, gt_situ_expos, cfg)

    # Feature selector (feature reduction)
    if cfg.PCA.STATE:
        reg_commu_features, gt_commu_expos = build_features(
                            trained_clusteror, commu_expo_features, gt_situ_expos, cfg)
        pca = PCA(n_components=cfg.PCA.N_COMPONENTS)
        pca.fit(reg_commu_features)
        X_train_rd = pca.transform(reg_train_features)

    else:
        pca = None
        X_train_rd = reg_train_features

    # Fit to the regressor
    trained_regressor = train_regressor(regressor, X_train_rd, gt_train_expos, cfg)

    return detectors, opt_threds, \
           trained_clusteror, trained_regressor, pca