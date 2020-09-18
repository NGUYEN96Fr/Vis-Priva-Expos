import os
from data.loader import data_loader
from modeling.builder import regressor_builder, clusteror_builder
from vispel.trainer import situ_trainer


class VISPEL:
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

    def eval_vispel(self):
        for situ_name, gt_situ_expos in self.gt_user_expos.items():
            if self.cfg.OUTPUT.VERBOSE:
                print(situ_name)
            # Evaluate ...
            situ_evaluator(self.clusterors[situ_name], self.regressors[situ_name], \
                           self.test_set, gt_situ_expos)