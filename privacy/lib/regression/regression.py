import random
import numpy as np
import scipy.stats as stats
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def parameter_search(data, tuned_parameters, scores):
    """

    Parameters
    ----------
    data: dict
        data to fine tune
    tuned_parameters: list
        list of parameters to fine tune
    scores: list
        criteria

    Returns
    -------

    """
    x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']
    print('Fine tuning for: ', 'pearson correlation')
    regr = GridSearchCV(SVR(), tuned_parameters)

    regr.fit(x_train, y_train)
    print('Best parameters:')
    print(regr.best_params_)
    print("Grid scores on development set:")
    means = regr.cv_results_['mean_test_score']
    stds = regr.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, regr.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    print("The scores are computed on the full evaluation set.")
    y_true, y_pred = y_test, regr.predict(x_test)
    print('pearson score: ', pear_corr(y_true, y_pred))


def train_regressor(x_train, y_train):
    """Train regressor by each situation

    :param: x_train: numpy array
        training data

    :param: y_train: numpy array
        training target


    return:
        trained model
        normalizer of training data

    """
    # regr = RandomForestRegressor(max_depth = max_depth, random_state= 0)
    regr = SVR(C =1, gamma =0.001, kernel ='rbf')
    regr.fit(x_train, y_train)

    return regr

def test_regressor(model, x_test, y_test):
    """Test regressor model

    :param: model
        trained model

    :param: normalizer
        data normalizer

    :param: fs
        feature selection

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

def train_test_split(regression_features, gt_expo_scores, train_ratio):
    """Split data into train and test set for a given situation

    :param: regression_features : dict
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
    random.seed(0)

    nb_users = len(list(gt_expo_scores.keys()))
    nb_user_not_consistent = 0
    X = []
    Y = []
    for user, score in gt_expo_scores.items():
        if user in regression_features:
            Y.append(score)
            X.append(regression_features[user])
        else:
            nb_user_not_consistent += 1

    X = np.asarray(X)
    Y = np.asarray(Y)

    indexes = np.linspace(0, nb_users - nb_user_not_consistent - 1, nb_users - nb_user_not_consistent, dtype=np.int32)
    random.shuffle(indexes)

    train_index = indexes[:int(nb_users * train_ratio)]
    test_index = indexes[int(nb_users * train_ratio):]

    situ_data = {'x_train': X[train_index, :], 'y_train': Y[train_index],
                 'x_test': X[test_index, :], 'y_test': Y[test_index]}

    print('     nb of users: ', nb_users - nb_user_not_consistent)
    print('     train profiles:', train_index.shape[0])

    return situ_data


def train_test_split_situ(regress_feature_situs, gt_user_expo_situs, train_ratio=0.8):
    """Train test split by situation

    :param: regress_feature_situs: dict
        user regression features in each situation
        {situ1: {user1: [feature1,...], ...}, ...}

    :param: gt_user_expo_situs: dict
        users and its ground truth crowd-sourcing user exposure scores in each situation
            {situ1: {user1: avg_score, ...}, ...}

    Returns
    -------
        train_test_situs: dict
            train and test data in each situation
                {situ1: {'x_train': ,'y_train': ,'x_test': ,'y_test': }, ...}
    """
    train_test_situs = {}

    for situ, gt_expo_user_scores in gt_user_expo_situs.items():
        print('  ', situ)
        train_test_situs[situ] = train_test_split(regress_feature_situs[situ], gt_expo_user_scores, train_ratio)

    return train_test_situs

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