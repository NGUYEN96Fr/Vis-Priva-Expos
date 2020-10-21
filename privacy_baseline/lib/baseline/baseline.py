import os
import json
from loader.loader import bloader
from situ.acronym import load_acronym, situ_decoding
from optimal_search.correlation import corr
from optimal_search.max_tau_subset import tau_subset, tau_max_cross_val
from optimal_search.optimal_thres_object import search_optimal_thres


class BASELINE(object):
    """
    Construct the baseline exposure prediction

    """

    def __init__(self, cfg, situ, save_file):
        self.cfg = cfg
        self.opt_thresholds = {}  # optimal threshold for each vis concept
        self.opt_detectors = {}  # selected detectors within its optimal thresholds
        self.test_result = 0.0
        self.save_file = save_file
        self.situ = situ_decoding(situ)
        self.root = os.getcwd().split('/privacy_baseline/tools')[0]
        self.save_path = os.path.join(self.cfg.OUTPUT.DIR, self.save_file.split('.pkl')[0] + '.txt')
        self.x_train, self.x_test, self.detectors, self.gt_expos = bloader(self.root, self.cfg, self.situ)

    def train(self):
        # optimal threshold for each vis concept
        self.opt_thresholds = search_optimal_thres(self.x_train, self.gt_expos, self.detectors,
                                                   self.cfg.SOLVER.CORR_TYPE, self.cfg)

        # optimal subset of vis concepts for each situ
        if self.cfg.SOLVER.CROSS_VAL:
            score_val_max, _, opt_detectors = tau_max_cross_val(self.x_train, self.gt_expos,
                                                                self.opt_thresholds, self.cfg.SOLVER.CORR_TYPE,
                                                                self.cfg, self.cfg.SOLVER.K_FOLDS)
            self.opt_detectors = opt_detectors

        else:
            tau_D_max, opt_detectors, _, _ = tau_subset(self.x_train, self.gt_expos,
                                                        self.opt_thresholds, self.cfg.SOLVER.CORR_TYPE, self.cfg)
            self.opt_detectors = opt_detectors

    def test(self):
        self.test_result = corr(self.x_test, self.gt_expos,
                                self.opt_detectors, self.cfg.SOLVER.CORR_TYPE, self.cfg, test_mode=True)

    def optimize(self):
        self.train()
        self.test()
