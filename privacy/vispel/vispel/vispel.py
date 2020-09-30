import os
from data.loader import data_loader
from vispel.trainer import situ_trainer
from exposure.exposure import community_expo
from regressor.features import build_features
from regressor.regression import test_regressor
from modeling.builder import regressor_builder, clusteror_builder
from clusteror.clustering import test_clusteror


class VISPEL(object):
    """
    Construct a end-to-end training pip-line for the VISPEL predictor

    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.root = os.getcwd().split('/privacy/tools')[0]
        self.X_mini_batches, self.X_test_set, self.X_community,\
        self.gt_user_expos,  self.vis_concepts = data_loader(self.root, self.cfg)
        self.clusterors = {}
        self.regressors = {}
        self.detectors = {}
        self.feature_selectors = {}
        self.opt_threds = {}
        self.test_results = {}

    def train_vispel(self):
        """

        :return:
        """
        if self.cfg.MODEL.DEBUG:
            self.X_train_set = self.X_mini_batches['30'] # 30 % training users
        else:
            self.X_train_set = self.X_mini_batches['100']


        print('Training clusteror, and regressor by situation ...')
        print('Eval mode: ',self.cfg.SOLVER.CORR_TYPE)

        for situ_name, gt_situ_expos in self.gt_user_expos.items():
            print(situ_name)
            # Initiate training models
            clusteror = clusteror_builder(self.cfg)
            regressor = regressor_builder(self.cfg)

            # Train ...
            detectors, opt_threds, trained_clusteror, trained_regressor, feature_selector = situ_trainer(situ_name, self.X_train_set, self.X_community,\
                                                        gt_situ_expos, self.vis_concepts, clusteror, regressor, self.cfg)
            self.clusterors[situ_name] = trained_clusteror
            self.regressors[situ_name] = trained_regressor
            self.detectors[situ_name] = detectors
            self.opt_threds[situ_name] = opt_threds
            self.feature_selectors[situ_name] = feature_selector

    def test_vispel(self):
        print("#-------------------------------------------------#")
        print("# Evaluate visual privacy exposure predictor       ")
        print("#-------------------------------------------------#")
        for situ_name, gt_situ_expos in self.gt_user_expos.items():
            print("****************************")
            print(situ_name)
            test_expo_features = community_expo(self.X_test_set, self.cfg.SOLVER.F_TOP, \
                                                 self.detectors[situ_name], self.opt_threds[situ_name], \
                                                 self.cfg.DETECTOR.LOAD, self.cfg, self.cfg.SOLVER.FILTERING)
            test_clusteror(situ_name, self.clusterors[situ_name], test_expo_features, self.cfg)
            reg_test_features, gt_test_expos = build_features(self.clusterors[situ_name],\
                                                    test_expo_features, gt_situ_expos, self.cfg)

            # Perform feature transform
            if self.cfg.PCA.STATE:
                X_test_rd = self.feature_selectors[situ_name].transform(reg_test_features)
                pca_variance = sum(self.feature_selectors[situ_name].explained_variance_ratio_)
            else:
                X_test_rd = reg_test_features
                pca_variance = None

            corr_score = test_regressor(self.regressors[situ_name],
                                        situ_name, X_test_rd, gt_test_expos, pca_variance, self.cfg)

            self.test_results[situ_name] = corr_score