import os
from sklearn.metrics import make_scorer
from user_situ_expos.user_expo import usr_photo_expo
from detectors.active import active_detectors
from clustering.features import clustering_photo_feature
from clustering.clustering import photo_user_expo_clustering
from regression.features import regression_features
from regression.regression import train_test_situs, train_regressor, test_regressor, pear_corr, kendall_corr, normalizer
from regression.fine_tuning import regress_fine_tuning

def parameter_search(root, gt_user_expo_situs, train_data, test_data, object_expo_situs, f_top, gamma, K, N, regm,
                     normalize, score_type, debug, feature_transform, load_active_detectors):
    """Searching best regression result for a current configuration

    :param root: string
        root working directory
    :param gt_user_expo_situs: dict
        ground-truth user exposure per situation
    :param train_data: dict
    :param test_data: dict
    :param object_expo_situs: dict
    :param f_top: float
        top  N_hat ranked detection
    :param gamma: float
        focusing exposure
    :param K: float
        scaling constant
    :param N: int
        number of clusters
    :param regm: string
        regression method
    :param: normalize: string
        if apply data normalization
    :param: debug mode
    :param: feature_transform: string
        apply feature transform function on photo features

    :return:
        best_result_situs : dict
            best result for all situations
            {situ1: {'reg_method': ,'corr_type': , 'best_params': ,'train_corr': ,'test_corr': ,'train_mse': ,'test_mse': }, ...}
    """

    ##Estimate exposure of user's photos in each situation
    print('Estimate exposure user photos ...')
    train_user_photo_expo_situs = {}
    test_user_photo_expo_situs = {}
    for situ_name, expo_clss in object_expo_situs.items():
        # activated detectors
        detectors, opt_threshs = active_detectors(expo_clss, situ_name, load_active_detectors)
        # estimate user's photo exposure
        train_user_photo_expo_situs[situ_name.split('.')[0]] = usr_photo_expo(train_data, f_top, detectors, opt_threshs, filter = True)
        test_user_photo_expo_situs[situ_name.split('.')[0]] = usr_photo_expo(test_data, f_top, detectors, opt_threshs, filter = True)
    print('Done!')

    ##Calculate clustering photo features
    print("#### CLUSTERING ####")

    print('Calculate clustering photo features ...')
    train_clustering_feature_situs = {}
    test_clustering_feature_situs = {}
    for situ_name, users in train_user_photo_expo_situs.items():
        train_clustering_feature_situs[situ_name] = clustering_photo_feature(situ_name, users, gamma, K)

    for situ_name, users in test_user_photo_expo_situs.items():
        test_clustering_feature_situs[situ_name] = clustering_photo_feature(situ_name, users, gamma, K)

    print('Done!')
    ##Photo clusters of each user per situation
    print('Calculate clusters of users ...')
    train_user_cluster_situations = {}
    test_user_cluster_situations ={}
    for situ_name, clustering_feature_users in train_clustering_feature_situs.items():
        train_user_cluster_situations[situ_name] = photo_user_expo_clustering(clustering_feature_users, N, feature_transform)

    for situ_name, clustering_feature_users in test_clustering_feature_situs.items():
        test_user_cluster_situations[situ_name] = photo_user_expo_clustering(clustering_feature_users, N, feature_transform)

    print('Done!')

    print("##### REGRESSION #####")
    print('Calculate regression features ...')
    train_regession_feature_situations = {}
    test_regession_feature_situations = {}
    for situ_name, user_clusters in train_user_cluster_situations.items():
        train_regession_feature_situations[situ_name] = regression_features(user_clusters)

    for situ_name, user_clusters in test_user_cluster_situations.items():
        test_regession_feature_situations[situ_name] = regression_features(user_clusters)
    print('Done!')

    print('Combine and convert to numpy format ...')
    train_test_batch_situs = train_test_situs(train_regession_feature_situations, test_regession_feature_situations, gt_user_expo_situs)
    print('Done!')

    best_result_situs = {}
    for situ, data in train_test_batch_situs.items():

        x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']
        if normalize:
            x_train, x_test = normalizer(x_train, x_test)
            data = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}

        print('********************************************')
        print(' ', situ)
        print('Searching best parameters by regressor ...')
        if not debug:

            if regm == 'svm':  # support vector machine
                tunning_parameters = {'kernel': ['rbf', 'linear', 'sigmoid'],
                                      'gamma': [1e-3, 1e-4, 1e-5],
                                      'C': [1, 5, 7, 10]}

            elif regm == 'rf':  # random forest
                tunning_parameters = {'bootstrap': [True, False],
                                      'max_depth': [3, 4, 5],
                                      'max_features': ['auto'],
                                      'min_samples_leaf': [1, 2, 4],
                                      'min_samples_split': [2, 3, 5],
                                      'n_estimators': [100, 150]}
        else:
            tunning_parameters = {'kernel': ['rbf', 'linear'],
                                  'gamma': [1e-3, 1e-4],
                                  'C': [1, 5]}

        if score_type == 'pear_corr':
            score = {score_type: make_scorer(pear_corr, greater_is_better=True)}

        elif score_type == 'kendall_corr':
            score = {score_type: make_scorer(kendall_corr, greater_is_better=True)}

        best_result = regress_fine_tuning(data, tunning_parameters, score, regm)
        best_result_situs[situ] = best_result

    print('Done!')

    return best_result_situs