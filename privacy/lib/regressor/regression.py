import os
import numpy as np
import scipy.stats as stats
from corr.corr_type import pear_corr, kendall_corr


def train_regressor(model, x_train, y_train, cfg):
    """Train regressor by each situation

    :param: x_train: numpy array
        training data

    :param: y_train: numpy array
        training target

    :return:
        trained modeling

    """

    if cfg.REGRESSOR.TYPE == 'RF':
        model.fit(x_train, y_train)

    if cfg.REGRESSOR.TYPE == 'SVM':
        model.fit(x_train, y_train)

    if cfg.OUTPUT.VERBOSE and cfg.FINE_TUNING.STATUS:
        print('Best fine_tuning parameters: ')
        print(model.best_params_)


    if cfg.OUTPUT.VERBOSE:
        y_pred = model.predict(x_train)

        # for k in range(y_train.shape[0]):
        #     print('gt= ',y_train[k],' pred= ',y_pred[k])

        if cfg.SOLVER.CORR_TYPE == 'KENDALL':
            print('correlation: ', kendall_corr(y_pred, y_train))
        elif cfg.SOLVER.CORR_TYPE == 'PEARSON':
            print('correlation: ', pear_corr(y_pred, y_train))

    return model

def test_regressor(model, x_test, y_test, cfg):
    """Train regressor by each situation

    :param: x_test: numpy array
        training data

    :param: y_test: numpy array
        training target

    :return:
        trained modeling

    """
    if cfg.REGRESSOR.TYPE == 'RF':
        y_pred = model.predict(x_test)

    if cfg.REGRESSOR.TYPE == 'SVM':
        y_pred = model.predict(x_test)

    # for k in range(y_test.shape[0]):
    #     print('gt= ',y_test[k],'pred= ',y_pred[k])
    if cfg.SOLVER.CORR_TYPE == 'KENDALL':
        corr = kendall_corr(y_pred, y_test)
        print('correlation: ', kendall_corr(y_pred, y_test))
    elif cfg.SOLVER.CORR_TYPE == 'PEARSON':
        corr = pear_corr(y_pred, y_test)
        print('correlation: ', pear_corr(y_pred, y_test))

    return corr