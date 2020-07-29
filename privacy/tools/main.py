import os
import sys
import  configparser
import _init_paths
from ablation_studies.focusing_exposure import gamma_study, gamma_plot


def ablation_study():

    ##get root directory
    root = os.path.dirname(os.path.dirname(os.getcwd()))
    #/home/users/vnguyen/intern20/Vis-Priva-Expos/
    ##read the param config file
    conf = configparser.ConfigParser()
    conf.read(sys.argv[1])
    conf = conf[os.path.basename(__file__)]

    gamma_path = 'out/abalation/gamma' ## abalation path
    gamma_file = 'gamma_rf_pear.json'
    normalize = False
    regm = 'svm' #regression method
    score_type = 'pear_corr' # or kendall_corr
    debug = True
    train_all = True
    only_plot = False

    if not only_plot:
        print('##########################################')
        print('######## Focusing Exposure Search ########')
        print('###########################################')
        gamma_list = [0, 3, 5, 7]
        gamma_study(gamma_list, conf, root, gamma_file, normalize, regm, score_type, debug, train_all)
        #gamma_plot(os.path.join(gamma_path), gamma_file)

    else:
        gamma_plot(os.path.join(gamma_path), gamma_file)

if __name__ == '__main__':
    ablation_study()
