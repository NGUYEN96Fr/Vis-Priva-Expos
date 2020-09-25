import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import f_regression
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
    :param X_community:
    :param gt_situ_expos:
        train user exposure
        in a given situation
    :param clusteror:
    :param regressor:
    :param cfg
    :param verbose:
    :return:
         trained cluster modeling for the situation
         trained regression modeling for the situation
    """
    # Construct active detectors
    detectors, opt_threds = activator(vis_concepts, situ_name,\
                                      cfg.DATASETS.PRE_VIS_CONCEPTS, cfg.DETECTOR.LOAD)

    # Photo exposures of users
    commu_expo_features = community_expo(X_community, cfg.SOLVER.F_TOP,\
                                       detectors, opt_threds, cfg.DETECTOR.LOAD, cfg, cfg.SOLVER.FILTERING)

    train_expo_features = community_expo(X_train_set, cfg.SOLVER.F_TOP,\
                                       detectors, opt_threds, cfg.DETECTOR.LOAD, cfg, cfg.SOLVER.FILTERING)

    # Build exposure features for users
    # by clustering their photo exposures
    trained_clusteror = train_clusteror(clusteror, commu_expo_features, cfg)

    # Build regression features for users
    reg_com_features, _ = build_features(trained_clusteror, commu_expo_features, gt_situ_expos, cfg)
    reg_train_features, gt_train_expos = build_features(trained_clusteror, train_expo_features, gt_situ_expos, cfg)

    # Feature selector
    pca = PCA(n_components=2)
    pca.fit(reg_com_features)
    print(pca.explained_variance_ratio_)
    X_train_rd = pca.transform(reg_train_features)
    # feature_selector = SelectKBest(score_func=f_regression, k=8)
    # feature_selector.fit(reg_train_features, gt_train_expos)
    # X_train_fs = feature_selector.transform(reg_train_features)

    # Fit to the regressor
    trained_regressor = train_regressor(regressor, X_train_rd, gt_train_expos, cfg)

    return detectors, opt_threds, \
           trained_clusteror, trained_regressor, pca