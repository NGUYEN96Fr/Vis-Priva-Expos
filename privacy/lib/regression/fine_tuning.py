from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from regression.regression import pear_corr

def regress_fine_tuning(data, tuned_parameters, scores, regm):
    """Estimate best parameters for regression

    Parameters
    ----------
    data: dict
        data to fine tune
    tuned_parameters: list
        list of parameters to fine tune
    scores: list
        criteria
    regm:
        regression method

    Returns
    -------
        best_result: dict
            {'reg_method': ,'corr_type': , 'best_params': ,'train_corr': ,'test_corr': ,'train_mse': ,'test_mse': }

    """

    best_result={}
    best_result['reg_method'] = regm
    best_result['corr_type'] = list(scores.keys())[0]

    x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']

    if regm == 'svm':
        regr = GridSearchCV(SVR(), tuned_parameters, cv=10, scoring=scores, refit=list(scores.keys())[0], n_jobs=2)
    elif regm == 'rf':
        regr = GridSearchCV(RFR(), tuned_parameters, cv=10, scoring=scores, refit=list(scores.keys())[0], n_jobs=2)

    regr.fit(x_train, y_train)
    print('Best parameters:')
    print(regr.best_params_)
    best_result['best_params'] = regr.best_params_

    print("Evaluation on train sets: ")
    y_true_train, y_pred_train = y_train,  regr.predict(x_train)
    print('Pearson score: ', pear_corr(y_true_train, y_pred_train))
    best_result['train_corr'] = pear_corr(y_true_train, y_pred_train)
    print('mse: ',  mean_squared_error(y_true_train, y_pred_train))
    best_result['train_mse'] = mean_squared_error(y_true_train, y_pred_train)

    print("Evaluation on test sets: ")
    y_true_test, y_pred_test = y_test, regr.predict(x_test)
    print('pearson score: ', pear_corr(y_true_test, y_pred_test))
    best_result['test_corr'] = pear_corr(y_true_test, y_pred_test)
    print('mse: ',mean_squared_error(y_true_test, y_pred_test))
    best_result['test_mse'] = mean_squared_error(y_true_test, y_pred_test)

    return best_result

