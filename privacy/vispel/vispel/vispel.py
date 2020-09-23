import os
from data.loader import data_loader
from vispel.trainer import situ_trainer
from exposure.exposure import community_expo
from regressor.features import build_features
from regressor.regression import test_regressor
from modeling.builder import regressor_builder, clusteror_builder


class VISPEL(object):
    """
    Construct a end-to-end training pip-line for the VISPEL predictor

    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.root = os.getcwd().split('/privacy/tools')[0]
        self.mini_batches, self.test_set, \
        self.gt_user_expos,  self.vis_concepts = data_loader(self.root, self.cfg)
        self.clusterors = {}
        self.regressors = {}
        self.detectors = {}
        self.opt_threds = {}
        self.test_results = {}


    def train_vispel(self):
        """

        :return:
        """
        if self.cfg.MODEL.DEBUG:
            self.train_set = self.mini_batches['30'] # 30 % training users
        else:
            self.train_set = self.mini_batches['100']

        if self.cfg.OUTPUT.VERBOSE:
            print('Training clusteror, and regressor by situation ...')
            print('Eval mode: ',self.cfg.SOLVER.CORR_TYPE)

        for situ_name, gt_situ_expos in self.gt_user_expos.items():
            if self.cfg.OUTPUT.VERBOSE:
                print(situ_name)
            # Initiate training models
            clusteror = clusteror_builder(self.cfg)
            regressor = regressor_builder(self.cfg)
            # Train ...
            detectors, opt_threds, trained_clusteror, trained_regressor = situ_trainer(situ_name, self.train_set,\
                                                        gt_situ_expos, self.vis_concepts, clusteror, regressor, self.cfg)
            self.clusterors[situ_name] = trained_clusteror
            self.regressors[situ_name] = trained_regressor
            self.detectors[situ_name] = detectors
            self.opt_threds[situ_name] = opt_threds

    def test_vispel(self):
        print("#-------------------------------------------------#")
        print("# Evaluate visual privacy exposure predictor       ")
        print("#-------------------------------------------------#")
        for situ_name, gt_situ_expos in self.gt_user_expos.items():
            print("***********************************************")
            print(situ_name)
            commu_expo_features = community_expo(self.test_set, self.cfg.SOLVER.F_TOP, \
                                                 self.detectors[situ_name], self.opt_threds[situ_name], \
                                                 self.cfg.DETECTOR.LOAD, self.cfg, self.cfg.SOLVER.FILTERING)

            reg_features, gt_expos = build_features(self.clusterors[situ_name],\
                                                    commu_expo_features, gt_situ_expos, self.cfg)
            corr = test_regressor(self.regressors[situ_name], reg_features, gt_expos, self.cfg)
            self.test_results[situ_name] = corr