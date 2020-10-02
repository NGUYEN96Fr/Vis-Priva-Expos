import os
import random
import numpy as np
from data.loader import data_loader
from situ.acronym import load_acronym
from vispel.trainer import trainer
from exposure.exposure import community_expo
from regressor.features import build_features
from regressor.regression import test_regressor
from modeling.builder import regressor_builder, clusteror_builder
from clusteror.clustering import test_clusteror


class VISPEL(object):
    """
    Construct a end-to-end training pip-line for the VISPEL predictor

    """

    def __init__(self, cfg, situation):
        self.cfg = cfg
        self.root = os.getcwd().split('/privacy/tools')[0]
        self.situation = situation
        self.situ_encoding = load_acronym(situation)
        self.X_train, self.X_test, self.X_community,\
                    self.gt_expos, self.vis_concepts, \
                    self.detectors,self.opt_threds= data_loader(self.root, self.cfg, self.situation)
        self.clusteror = None
        self.regressor = None
        self.feature_selector = None
        self.test_result = None
        self.set_seeds()

    def set_seeds(self):
        random.seed(self.cfg.MODEL.SEED)
        np.random.seed(self.cfg.MODEL.SEED)

    def train_vispel(self):
        """

        :return:
        """
        if self.cfg.OUTPUT.VERBOSE:
            print("#-------------------------------------------------#")
            print("# Train visual privacy exposure predictor          ")
            print("#                  %s          " %self.situ_encoding)
            print("#-------------------------------------------------#")

        # Initiate training models
        clusteror = clusteror_builder(self.cfg)
        regressor = regressor_builder(self.cfg)

        # Train ...
        trained_clusteror, trained_regressor, feature_selector = trainer(self.situation, self.X_train, self.X_community,\
                                                    self.gt_expos, clusteror, regressor, self.detectors, self.opt_threds, self.cfg)

        self.clusteror = trained_clusteror
        self.regressor = trained_regressor
        self.feature_selector = feature_selector

    def test_vispel(self):

        if self.cfg.OUTPUT.VERBOSE:
            print("#-------------------------------------------------#")
            print("# Evaluate visual privacy exposure predictor       ")
            print("#-------------------------------------------------#")

        test_expo_features = community_expo(self.X_test, self.cfg.SOLVER.F_TOP,\
                                            self.detectors, self.opt_threds, self.cfg.DETECTOR.LOAD,
                                            self.cfg, self.cfg.SOLVER.FILTERING)

        reg_test_features, gt_test_expos = build_features(self.clusteror, test_expo_features,
                                                          self.gt_expos, self.cfg)

        test_clusteror(self.situation, self.clusteror, test_expo_features, self.cfg)

        # Perform feature transform
        if self.cfg.PCA.STATE:
            X_test_rd = self.feature_selector.transform(reg_test_features)
            pca_variance = sum(self.feature_selector.explained_variance_ratio_)
        else:
            X_test_rd = reg_test_features
            pca_variance = 0

        corr_score = test_regressor(self.regressor, self.situation,
                                    X_test_rd, gt_test_expos, pca_variance, self.cfg)

        self.test_result = corr_score