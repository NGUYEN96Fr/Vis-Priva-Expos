import os
import json
from loader.loader import bloader
from situ.acronym import load_acronym, situ_decoding
from optimal_search.correlation import corr
from optimal_search.optimal_thres import search_optimal_thres


class SBASELINE(object):
    """
    Construct the baseline exposure prediction

    """

    def __init__(self, cfg, situ, save_file):
        self.cfg = cfg
        self.test_result = 0.0
        self.opt_threshold = 0.0
        self.save_file = save_file
        self.situ = situ_decoding(situ)
        self.root = os.getcwd().split('/sbaseline/tools')[0]
        self.save_path = os.path.join(self.cfg.OUTPUT.DIR, self.save_file.split('.pkl')[0] + '.txt')
        self.x_train, self.x_test, self.detectors, self.gt_expos = bloader(self.root, self.cfg, self.situ)

    def train(self):
        # optimal threshold for each vis concept
        self.opt_threshold = search_optimal_thres(self.x_train, self.gt_expos, self.detectors,
                                                  self.cfg.SOLVER.CORR_TYPE, self.cfg)

    def test(self):
        tdetectors = {}
        for detector, score in self.detectors.items():
            tdetectors[detector] = [self.opt_threshold, score]

        self.test_result = corr(self.x_test, self.gt_expos,
                                tdetectors, self.cfg.SOLVER.CORR_TYPE, self.cfg)

    def optimize(self):
        self.train()
        self.test()
