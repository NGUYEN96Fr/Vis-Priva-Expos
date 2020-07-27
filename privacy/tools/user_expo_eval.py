import os
import sys
import  configparser
import _init_paths
# import seaborn as sns
# import scipy
# import numpy as np
# import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import make_scorer

from user_imgs.retrieve import retrieve_detected_objects, retrieve_photos
from situations.load_situs import load_situs
from user_situ_expos.user_expo import _photos_users
from detectors.active import active_detectors
from clustering.features import clustering_photo_feature
from clustering.clustering import photo_user_expo_clustering
from regression.features import regression_features
from regression.regression import train_test_split_situ, train_regressor, test_regressor, parameter_search, pear_corr, normalizer
from preprocess.user import load_gt_user_profiles

def main():

    ##get root directory
    root = os.path.dirname(os.getcwd())

    ##read the param config file
    conf = configparser.ConfigParser()
    conf.read(sys.argv[1])
    conf = conf[os.path.basename(__file__)]

    ##get params
    inference_file = conf['inference_file']
    siutation_file = conf['situation_file']
    user_profile_path = conf['user_profile_path']
    f_top = float(conf['f_top'])
    K = float(conf['K'])
    gamma = float(conf['gamma'])
    N = int(conf['N'])
    train_ratio = float(conf['train_ratio'])
    normalize = False
    fine_tuning = True

    ##Load crowdsourcing user privacy exposure scores in each situation
    gt_user_expo_situs = load_gt_user_profiles(os.path.join(root,user_profile_path))

    ##Read user's photos
    objects_photo_per_user = retrieve_detected_objects(os.path.join(root, inference_file))

    ##Read object exposures in each situation
    object_expo_situs = load_situs(os.path.join(root, siutation_file))

    ##Estimate exposure of user's photos in each situation
    print('Estimate exposure user photos ...')
    user_photo_expo_situs = {}
    for situ_name, expo_clss in object_expo_situs.items():
        #activated detectors
        detectors = active_detectors(expo_clss)
        #estimate user's photo exposure
        user_photo_expo_situs[situ_name.split('.')[0]] = _photos_users(objects_photo_per_user,f_top,detectors)
    print('Done!')

    ##Calculate clustering photo features
    print("####################")
    print("#### CLUSTERING ####")
    print("####################")

    print('Calculate clustering photo features ...')
    clutering_feature_situs = {}
    for situ_name, users in user_photo_expo_situs.items():
        clutering_feature_situs[situ_name] = clustering_photo_feature(situ_name,users,gamma,K)

    print('Done!')
    ##Photo clusters of each user per situation
    print('Calculate clusters of users ...')
    user_cluster_situations = {}
    for situ_name, clustering_feature_users in clutering_feature_situs.items():
        user_cluster_situations[situ_name] = photo_user_expo_clustering(clustering_feature_users, N)

    print('Done!')

    print("######################")
    print("##### REGRESSION #####")
    print("######################")
    print('Calculate regression features ...')
    regession_feature_situations = {}
    for situ_name, user_clusters in user_cluster_situations.items():
        regession_feature_situations[situ_name] = regression_features(user_clusters)
    print('Done!')

    print('Split into train and test sets ...')
    train_test_situs = train_test_split_situ(regession_feature_situations,gt_user_expo_situs,train_ratio)
    print('Done!')
    
    print('Train and test regressor by situation ...')
    for situ, data in train_test_situs.items():

        x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']
        if normalize:
            x_train, x_test = normalizer(x_train, x_test)
            data = {'x_train': x_train, 'y_train': y_train, 'x_test':x_test, 'y_test':y_test}

        if not fine_tuning:
            print(' ',situ)
            print('Training...')
            model = train_regressor(x_train, y_train)
            print('Testing...')
            # test_regressor(model, x_train, y_train)
            test_regressor(model, x_test, y_test)

        else:
            print(' ', situ)
            print('Fine Tuning ...')
            tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-5],
                                 'C': [1, 10, 100, 1000]},
                                {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
            scores = {'pear_corr': make_scorer(pear_corr)}
            parameter_search(data, tuned_parameters, scores)


    print('Done!')


if __name__ == '__main__':
    main()
