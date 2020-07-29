import os
import json
import numpy as np
import matplotlib.pyplot as plt
from ablation_studies.param_search import parameter_search
from loader.data_loader import load_train_test, load_gt_user_expo, load_situs

def gamma_study(gamma_list, conf, root, gamma_file, normalize, regm, score_type, debug, train_all):
    """Study impact of the gamma parameter

    :param conf: current configuration
    :param gamma_list: list of gamma values
    :param root: current working directory
    :param gamma_file: gamma saved file name
    :param normalize: if normalize regression features
    :param regm: regression method
    :param score_type: score type (kendall or pearson)
    :param train_all: if use all training data
    :param debug: turn on debug mode

    :return:

    """
    ##get params
    train_test_path = conf['train_test_path']
    gt_expo_path = conf['gt_expo_path']
    siutation_file = conf['situation_path']
    outdir = conf['outdir']
    f_top = float(conf['f_top'])
    K = float(conf['K'])
    N = int(conf['N'])

    ##Load crowdsourcing user privacy exposure scores in each situation
    gt_user_expo_situs = load_gt_user_expo(root, gt_expo_path)
    ##Load train and test data
    minibatches, test_data = load_train_test(root, train_test_path)
    ##Read object exposures in each situation
    object_expo_situs = load_situs(root, siutation_file)

    if train_all:
        gamma_search = {}
        train_data = minibatches['100']

        for gamma in gamma_list:
            gamma_search[gamma] = parameter_search(root, gt_user_expo_situs, train_data, test_data, object_expo_situs, f_top, gamma, K, N, regm,
                         normalize, score_type, debug)

        abalation_dir = os.path.join(root, outdir, 'abalation')
        if not os.path.exists(abalation_dir):
            os.makedirs(abalation_dir)

        save_dir = os.path.join(root, abalation_dir, 'gamma')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        ## save file
        print('Saving files ...')
        with open(os.path.join(save_dir, gamma_file), 'w') as fp:
            json.dump(gamma_search, fp)


def gamma_plot(gamma_path, gamma_file):
    """
    :param gamma_path: string
        path to gamma abalation study
        {gamma1: {...}, ...}
        {...} = {situ1: {'reg_method': ,'corr_type': ,'best_params': ,'train_corr': ,'test_corr': ,'train_mse': ,'test_mse': }, ...}

    """
    gamma_search = json.load(open(os.path.join(gamma_path, gamma_file)))
    print('************************')
    print('**** Plot gamma ********')
    print('************************')

    fig, ax = plt.subplots(2, 2)
    width = 0.3
    nb_gammas = len(list(gamma_search.keys()))
    x = np.arrange(4) #4 situation
    k = -1 # index to plot


    for gamma, situs in gamma_search.items():
        train_corr_erros = []
        test_corr_erros = []
        train_mse_errors = []
        test_mse_errors = []
        situ_labels = []
        k += 1
        for situ, results in situs.items():
            if situ not in situ_labels:
                situ_labels.append(situ)

            train_corr_erros.append(results['train_corr'])
            test_corr_erros.append(results['test_corr'])
            train_mse_errors.append(results['train_mse'])
            test_mse_errors.append(results['test_mse'])

            ax[0][0].bar(x + (2 * k - 1) / 2 * width, train_corr_erros, width, label=r'$\gamma$'+'='+str(gamma))
            ax[0][1].bar(x + (2 * k - 1) / 2 * width, train_mse_errors, width, label=r'$\gamma$'+'='+str(gamma))
            ax[1][0].bar(x + (2 * k - 1) / 2 * width, test_corr_erros, width, label=r'$\gamma$'+'='+str(gamma))
            ax[1][1].bar(x + (2 * k - 1) / 2 * width, test_mse_errors, width, label=r'$\gamma$'+'='+str(gamma))

    ax[0][0].set_ylabel('correlation')
    ax[1][0].set_ylabel('correlation')
    ax[0][1].set_ylabel('mse')
    ax[1][1].set_ylabel('mse')

    ax[0][0].set_xticks(x)
    ax[0][0].set_xticklabels(situ_labels)
    ax[0][1].set_xticks(x)
    ax[0][1].set_xticklabels(situ_labels)
    ax[1][0].set_xticks(x)
    ax[1][0].set_xticklabels(situ_labels)
    ax[1][1].set_xticks(x)
    ax[1][1].set_xticklabels(situ_labels)

    ax[0][0].set_title('Train Correlation')
    ax[1][0].set_title('Test Correlation')
    ax[0][1].set_title('Train MSE')
    ax[1][1].set_title('Test MSE')

    ax[0][0].legend()
    ax[1][0].legend()
    ax[0][1].legend()
    ax[1][1].legend()

    fig.tight_layout()

    plt.show()
    plt.savefig(gamma_file.split('.')[0]+'.png')
