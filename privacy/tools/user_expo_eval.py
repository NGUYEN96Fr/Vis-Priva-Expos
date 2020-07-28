import os
import sys
import  configparser
import _init_paths
from ablation_studies.param_search import parameter_search


def ablation_study():

    ##get root directory
    root = os.path.dirname(os.getcwd())

    ##read the param config file
    conf = configparser.ConfigParser()
    conf.read(sys.argv[1])
    conf = conf[os.path.basename(__file__)]

    ##get params
    inference_file = conf['inference_file']
    siutation_file = conf['situation_file']
    user_profile_path = conf['user_profile_path']
    f_top = float(conf['f_top'])
    K = float(conf['K'])
    gamma = float(conf['gamma'])
    N = int(conf['N'])
    train_ratio = float(conf['train_ratio'])
    normalize = False
    regm = 'svm' #regression method

    best_result_situs = parameter_search(root, user_profile_path, inference_file, siutation_file, f_top, gamma, K, N, train_ratio, regm,
                     normalize)


if __name__ == '__main__':
    ablation_study()
