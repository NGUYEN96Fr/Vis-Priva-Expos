import random
import numpy as np
import scipy.stats as stats
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.svm import SVR
from sklearn.metrics import r2_score


def train_regressor(x_train, y_train, params, regm):
    """Train regressor by each situation

    :param: x_train: numpy array
        training data

    :param: y_train: numpy array
        training target

    :param: params: dict
        parameter to train model

    :param: regm: string
        regression method

    :return:
        trained model

    """

    if regm == 'svm':
        regr = SVR(kernel=params['kernel'], C=params['C'], gamma=params['gamma'])

    elif regm == 'rf':
        regr = RFR(bootstrap=params['bootstrap'], max_depth=params['max_depth'], max_features =params['max_features'],
                   min_samples_leaf=params['min_samples_leaf'],min_samples_split=params['min_samples_split'],n_estimators=params['n_estimators'])
    regr.fit(x_train, y_train)

    return regr

def test_regressor(model, x_test, y_test):
    """Test regressor model

    :param: model
        trained model

    :param: normalizer
        data normalizer

    :param: x_test
        test data

    :param: y_test
        test target

    """
    y_pred = model.predict(x_test)
    y_true = y_test

    for i in range(x_test.shape[0]):
        print('gt = ', y_true[i], ' prediction =', y_pred[i])

    r, _ = stats.pearsonr(y_true, y_pred)
    print('Pearson correlation = ', r)


def normalizer(x_train, x_test):
    """

    Parameters
    ----------
    x_train
    x_test

    Returns
    -------
        normalized data-set
    """
    mean_x = np.mean(x_train, axis=0)
    std_x = np.std(x_train, axis=0)

    x_train_normalized = np.divide(x_train - mean_x, std_x)
    x_test_normalized = np.divide(x_test - mean_x, std_x)

    return x_train_normalized, x_test_normalized

def train_test_combine(train_regression_features,test_regression_features, gt_expo_scores):
    """Combine train and test sets into a dict for a given situation

    :param: train_regression_features : dict
        indiviual user and its feature
            {user1: [feature1,...], ...}

    :param: test_regression_features : dict
        indiviual user and its feature
            {user1: [feature1,...], ...}

    :param: gt_expo_scores : dict
        user and its ground truth crowd-sourcing user exposure scores
            {user1: avg_score, ...}
    Returns
    -------
        situ_data : dict

            with following fields

                X_train, X_test : numpy array
                    (N, #features)

                Y_train, Y_test : numpy array
                    (N, )
    """

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for user, features in train_regression_features.items():
        x_train.append(features)
        y_train.append(gt_expo_scores[user])

    for user, features in test_regression_features.items():
        x_test.append(features)
        y_test.append(gt_expo_scores[user])

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    situ_data = {'x_train': x_train, 'y_train': y_train,
                 'x_test': x_test, 'y_test': y_test}

    print('Number of training data: ', x_train.shape[0])
    print('Number of test data: ', x_test.shape[0])

    return situ_data


def train_test_situs(train_regession_feature_situations, test_regession_feature_situations, gt_user_expo_situs):
    """Train test data by situation

    :param: train_regession_feature_situations: dict
        training regression features in each situation
            {situ1: {user1: [feature1,...], ...}, ...}

    :param: test_regession_feature_situations: dict
        testing regression features in each situation
            {situ1: {user1: [feature1,...], ...}, ...}

    :param: gt_user_expo_situs: dict
        users and its ground truth crowd-sourcing user exposure scores in each situation
            {situ1: {user1: avg_score, ...}, ...}

    Returns
    -------
        train_test: dict
            train and test data in each situation
                {situ1: {'x_train': ,'y_train': ,'x_test': ,'y_test': }, ...}
    """
    train_test= {}

    for situ, gt_expo_user_scores in gt_user_expo_situs.items():
        print('  ', situ)
        train_test[situ] = train_test_combine(train_regession_feature_situations[situ],
                                            test_regession_feature_situations[situ],
                                            gt_expo_user_scores)

    return train_test


def pear_corr(y_true, y_pred):
    """Calculate pearson correlation

    Parameters
    ----------
    y_true
    y_pred

    Returns
    -------
        r : float
            correlation value
    """
    r, _ = stats.pearsonr(y_true,y_pred)
    return r


def kendall_corr(y_true, y_pred):
    """Calculate pearson correlation

    Parameters
    ----------
    y_true
    y_pred

    Returns
    -------
        r : float
            correlation value
    """
    r, _ = stats.kendalltau(y_true,y_pred)

    return r