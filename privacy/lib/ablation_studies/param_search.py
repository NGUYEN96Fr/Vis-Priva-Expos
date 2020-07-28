import os
from sklearn.metrics import make_scorer
from user_imgs.retrieve import retrieve_detected_objects, retrieve_photos
from situations.load_situs import load_situs
from user_situ_expos.user_expo import _photos_users
from detectors.active import active_detectors
from clustering.features import clustering_photo_feature
from clustering.clustering import photo_user_expo_clustering
from regression.features import regression_features
from regression.regression import train_test_split_situ, train_regressor, test_regressor, pear_corr, kendall_corr, normalizer
from regression.fine_tuning import regress_fine_tuning
from preprocess.user import load_gt_user_profiles

def parameter_search(root, user_profile_path, inference_file, siutation_file, f_top, gamma, K, N, train_ratio, regm, normalize, debug = True):
    """Searching best regression result for a current configuration

    :param root: string
        root working directory
    :param user_profile_path: string
        path to user exposure files
    :param inference_file: string
        path to object detection inference on the user profiles
    :param siutation_file: string
        path to object-dependent exposures for situations
    :param f_top: float
        top  N_hat ranked detection
    :param gamma: float
        focusing exposure
    :param K: float
        scaling constant
    :param N: int
        number of clusters
    :param train_ratio: float
        percentage of training data
    :param regm: string
        regression method
    :param: normalize: string
        if apply data normalization
    :param: debug mode

    :return:
        best_result_situs : dict
            best result for all situations
            {situ1: {'reg_method': ,'corr_type': , 'best_params': ,'train_corr': ,'test_corr': ,'train_mse': ,'test_mse': }, ...}
    """

    ##Load crowdsourcing user privacy exposure scores in each situation
    gt_user_expo_situs = load_gt_user_profiles(os.path.join(root, user_profile_path))

    ##Read user's photos
    objects_photo_per_user = retrieve_detected_objects(os.path.join(root, inference_file))

    ##Read object exposures in each situation
    object_expo_situs = load_situs(os.path.join(root, siutation_file))

    ##Estimate exposure of user's photos in each situation
    print('Estimate exposure user photos ...')
    user_photo_expo_situs = {}
    for situ_name, expo_clss in object_expo_situs.items():
        # activated detectors
        detectors = active_detectors(expo_clss)
        # estimate user's photo exposure
        user_photo_expo_situs[situ_name.split('.')[0]] = _photos_users(objects_photo_per_user, f_top, detectors)
    print('Done!')

    ##Calculate clustering photo features
    print("#### CLUSTERING ####")

    print('Calculate clustering photo features ...')
    clutering_feature_situs = {}
    for situ_name, users in user_photo_expo_situs.items():
        clutering_feature_situs[situ_name] = clustering_photo_feature(situ_name, users, gamma, K)

    print('Done!')
    ##Photo clusters of each user per situation
    print('Calculate clusters of users ...')
    user_cluster_situations = {}
    for situ_name, clustering_feature_users in clutering_feature_situs.items():
        user_cluster_situations[situ_name] = photo_user_expo_clustering(clustering_feature_users, N)

    print('Done!')

    print("##### REGRESSION #####")
    print('Calculate regression features ...')
    regession_feature_situations = {}
    for situ_name, user_clusters in user_cluster_situations.items():
        regession_feature_situations[situ_name] = regression_features(user_clusters)
    print('Done!')

    print('Split into train and test sets ...')
    train_test_situs = train_test_split_situ(regession_feature_situations, gt_user_expo_situs, train_ratio)
    print('Done!')

    best_result_situs = {}
    for situ, data in train_test_situs.items():

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
                                      'max_depth': [3, 5, 7],
                                      'max_features': ['auto', 'sqrt'],
                                      'min_samples_leaf': [1, 2, 4],
                                      'min_samples_split': [2, 3, 5],
                                      'n_estimators': [100, 130, 160]}
        else:
            tunning_parameters = {'bootstrap': [True],
                                  'max_depth': [3, 5, 7],
                                  'max_features': ['auto'],
                                  'min_samples_leaf': [4],
                                  'min_samples_split': [2],
                                  'n_estimators': [100]}

        scores = {'pear_corr': make_scorer(pear_corr, greater_is_better=True)}
        best_result = regress_fine_tuning(data, tunning_parameters, scores, regm)
        best_result_situs[situ] = best_result

    print('Done!')

    return best_result_situs