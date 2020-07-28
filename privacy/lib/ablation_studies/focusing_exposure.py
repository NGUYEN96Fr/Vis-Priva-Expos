import os
import json
import numpy as np
import matplotlib.pyplot as plt
from ablation_studies.param_search import parameter_search

def gamma_study(gamma_list, conf, root, gamma_file):
    """Study impact of the gamma parameter

    :param conf: current configuration
    :param gamma_list: list of gamma values
    :param root: current working directory
    :param gamma_file: gamma saved file

    :return:

    """
    ##get params
    inference_file = conf['inference_file']
    siutation_file = conf['situation_file']
    user_profile_path = conf['user_profile_path']
    outdir = conf['outdir']
    f_top = float(conf['f_top'])
    K = float(conf['K'])
    N = int(conf['N'])
    train_ratio = float(conf['train_ratio'])
    normalize = False
    regm = 'rf' #regression method
    debug = True

    gamma_search = {}
    for gamma in gamma_list:
        gamma_search[gamma] = parameter_search(root, user_profile_path, inference_file, siutation_file, f_top, gamma, K, N, train_ratio, regm,
                     normalize, debug)

    abalation_dir = os.path.join(outdir,'abalation')
    if not os.path.exists(abalation_dir):
        os.makedirs(abalation_dir)

    save_dir = os.path.join(abalation_dir,'gamma')
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