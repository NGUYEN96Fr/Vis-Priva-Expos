"""


"""
import os
import numpy as np
import pickle
from exposure.exposure import community_expo
from detectors.activator import activator
from clusteror.clustering import train_clusteror
from regressor.features import build_cnn_features
from data.raw_loader import raw_data_loader
from modeling.builder import clusteror_builder


class GData(object):
    """
    Generate train data, and test data for training CNN models

    """
    def __init__(self, cfg, file_name):
        self.cfg = cfg
        self.file_name = file_name
        self.root = os.getcwd().split('/privacy/tools')[0]
        self.mini_batches, self.test_set, \
        self.gt_user_expos, self.vis_concepts = raw_data_loader(self.root, self.cfg)
        self.clusterors = {}
        self.detectors = {}
        self.opt_threds = {}
        self.cnn_train = {}
        self.cnn_train['X_features'] = {}
        self.cnn_train['y_targets'] = {}
        self.cnn_test = {}
        self.cnn_test['X_features'] = {}
        self.cnn_test['y_targets'] = {}

    def train_data(self):
        """
        Generate training data

        :return:
        """
        print('Generating train data ...')
        self.train_set = self.mini_batches['100']

        for situ_name, gt_situ_expos in self.gt_user_expos.items():
            if self.cfg.OUTPUT.VERBOSE:
                print(situ_name)
            # Initiate training models
            clusteror = clusteror_builder(self.cfg)

            # Construct active detectors
            detectors, opt_threds = activator(self.vis_concepts, situ_name, \
                                              self.cfg.DATASETS.PRE_VIS_CONCEPTS, self.cfg.DETECTOR.LOAD)

            # Photo exposures of users
            commu_expo_features = community_expo(self.train_set, self.cfg.SOLVER.F_TOP, \
                                                 detectors, opt_threds, self.cfg.DETECTOR.LOAD, self.cfg.SOLVER.FILTERING)

            # Build exposure features for users
            # by clustering their photo exposures
            trained_clusteror = train_clusteror(clusteror, commu_expo_features, self.cfg)

            for user, user_expo_features in commu_expo_features.items():
                cnn_features = build_cnn_features(trained_clusteror, user_expo_features, self.cfg)

                if user not in  self.cnn_train['X_features']:
                    self.cnn_train['X_features'][user] = cnn_features
                    self.cnn_train['y_targets'][user] = np.asarray(gt_situ_expos[user]).reshape(1, 1, 1)
                else:

                    self.cnn_train['X_features'][user] = np.concatenate((self.cnn_train['X_features'][user],cnn_features), axis=1)
                    self.cnn_train['y_targets'][user] = np.concatenate((self.cnn_train['y_targets'][user],\
                                                                       np.asarray(gt_situ_expos[user]).reshape(1, 1, 1)), axis=1)
            self.clusterors[situ_name] = trained_clusteror
            self.detectors[situ_name] = detectors
            self.opt_threds[situ_name] = opt_threds

    def test_data(self):
        """

        :return:
        """
        print('Generating test data ...')
        for situ_name, gt_situ_expos in self.gt_user_expos.items():
            if self.cfg.OUTPUT.VERBOSE:
                print(situ_name)
            commu_expo_features = community_expo(self.test_set, self.cfg.SOLVER.F_TOP, \
                                                 self.detectors[situ_name], self.opt_threds[situ_name], \
                                                 self.cfg.DETECTOR.LOAD, self.cfg.SOLVER.FILTERING)

            for user, user_expo_features in commu_expo_features.items():
                cnn_features = build_cnn_features(self.clusterors[situ_name], user_expo_features, self.cfg)
                if user not in self.cnn_test['X_features']:
                    self.cnn_test['X_features'][user] = cnn_features
                    self.cnn_test['y_targets'][user] = np.asarray(gt_situ_expos[user]).reshape(1,1,1)
                else:
                    self.cnn_test['X_features'][user] = np.concatenate((self.cnn_test['X_features'][user],cnn_features), axis=1)
                    self.cnn_test['y_targets'][user] = np.concatenate((self.cnn_test['y_targets'][user],\
                                                                       np.asarray(gt_situ_expos[user]).reshape(1,1,1)), axis=1)
    def save(self):
        """
        data saver

        :return:
        """
        self.train_data()
        self.test_data()

        CNN_DATA = {}
        CNN_DATA['train'] = {}
        CNN_DATA['test'] = {}
        init_dict = True

        for user, X_features in self.cnn_train['X_features'].items():

            if init_dict:
                CNN_DATA['train']['features'] = X_features
                CNN_DATA['train']['expos'] = self.cnn_train['y_targets'][user]
                init_dict = False
            else:
                CNN_DATA['train']['features'] = np.concatenate((CNN_DATA['train']['features'], X_features), axis= 0)
                CNN_DATA['train']['expos'] = np.concatenate((CNN_DATA['train']['expos'],  self.cnn_train['y_targets'][user]), axis= 0)

        init_dict = True
        for user, X_features in self.cnn_test['X_features'].items():

            if init_dict:
                CNN_DATA['test']['features'] = X_features
                CNN_DATA['test']['expos'] = self.cnn_test['y_targets'][user]
                init_dict = False
            else:
                CNN_DATA['test']['features'] = np.concatenate((CNN_DATA['test']['features'], X_features), axis=0)
                CNN_DATA['test']['expos'] = np.concatenate((CNN_DATA['test']['expos'],
                                                            self.cnn_test['y_targets'][user]), axis=0)

        print('#------------------#')
        print('#CNN DATA')
        print('#------------------#')
        print('Train Data')
        print(CNN_DATA['train'].keys())
        print('Feature Shapes: ', CNN_DATA['train']['features'].shape)
        print('Expo Shapes: ', CNN_DATA['train']['expos'].shape)
        print('Test Data')
        print('Feature Shapes: ', CNN_DATA['test']['features'].shape)
        print('Expo Shapes: ', CNN_DATA['test']['expos'].shape)

        root = os.getcwd()
        out_dir_path = os.path.join(root, self.cfg.OUTPUT.DIR)
        if not os.path.exists(out_dir_path):
            os.makedirs(out_dir_path)

        out_file_path = os.path.join(out_dir_path, self.file_name)

        with open(out_file_path, 'wb') as output:
            pickle.dump(CNN_DATA, output, pickle.HIGHEST_PROTOCOL)

        return CNN_DATA['train'], CNN_DATA['test']