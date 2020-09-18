import os
import numpy as np
from joblib import dump
import scipy.stats as stats
from situ.acronym import load_acronym
from corr.corr_type import pear_corr, kendall_corr


def train_regressor(situ_name, model, x_train, y_train, cfg):
    """Train regressor by each situation

    :param: x_train: numpy array
        training data

    :param: y_train: numpy array
        training target

    :return:
        trained model

    """
    ACRONYMS = load_acronym()

    print(x_train.shape)
    print(y_train.shape)

    if cfg.REGRESSOR.TYPE == 'RF':
        model.fit(x_train, y_train)

    if cfg.REGRESSOR.TYPE == 'SVM':
        model.fit(x_train, y_train)

    if cfg.OUTPUT.VERBOSE and cfg.FINE_TUNING.STATUS:
        print('Best fine_tuning parameters: ')
        print(model.best_params_)

    save_path = os.path.join(cfg.OUTPUT.DIR, 'regressor_'+ACRONYMS[situ_name]+'.joblib')
    if cfg.OUTPUT.VERBOSE:
        y_pred = model.predict(x_train)
        if cfg.SOLVER.CORR_TYPE == 'KENDALL':
            print('accuracy: ', kendall_corr(y_pred, y_train))
        elif cfg.SOLVER.CORR_TYPE == 'PEARSON':
            print('accuracy: ', pear_corr(y_pred, y_train))
        print('Save regressor model !!! ')

    dump(model, save_path)

    return model