import os
import json
from param_search import parameter_search

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
    regms = ['svm', 'rf'] #regression method

    gamma_search = {}
    for gamma in gamma_list:
        gamma_search[gamma] = {}
        for regm in regms:
            gamma_search[gamma][regm] = parameter_search(root, user_profile_path, inference_file, siutation_file, f_top, gamma, K, N, train_ratio, regm,
                         normalize)

    abalation_dir = os.path.join(outdir,'abalation')
    if not os.path.exists(abalation_dir):
        os.makedirs(abalation_dir)

    save_dir = os.path.join(abalation_dir,'gamma')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ## save file
    with open(os.path.join(save_dir, gamma_file), 'w') as fp:
        json.dumps(gamma_search, fp)


def gamma_plot(gamma_path, gamma_file):
    """
    :param gamma_path: string
        path to gamma abalation study

    """
    gamma_search = json.load(open(os.path.join(gamma_path,gamma_file)))
    print('************************')
    print('**** Plot gamma ********')
    print('************************')
    print(gamma_search)