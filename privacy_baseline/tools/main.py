import os
import sys
import numpy as np
import configparser
import _init_paths
import json
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from optimal_search.max_tau_subset import tau_subset, tau_max_cross_val
from optimal_search.optimal_thres_object import search_optimal_thres, export_tau_ranking
from optimal_search.correlation import corr, pos_neg_corr
from loader.data_loader import load_train_test, load_gt_user_expo, load_situs

def main():
    """

    """
    train_all = True
    ##get root directory
    root = os.path.dirname(os.path.dirname(os.getcwd()))
    ##get params
    conf = configparser.ConfigParser()
    conf.read(sys.argv[1])
    conf = conf[os.path.basename(__file__)]
    
    train_test_path = conf['train_test_path']
    gt_expo_path = conf['gt_expo_path']
    situation_file = conf['situation_path']
    outdir = conf['outdir']
    if not os.path.exists(os.path.join(root, 'privacy_baseline', outdir)):
        os.makedirs(os.path.join(root, 'privacy_baseline', outdir))
    
    corr_type = 'kendall_corr'
    debug = False
    load = False
    plot_ = False
    cross_val = False
    save_file = '3_'

    ##Load crowdsourcing user privacy exposure scores in each situation
    gt_user_expo_situs = load_gt_user_expo(root, gt_expo_path)
    ##Load train and test data
    minibatches, test_data = load_train_test(root, train_test_path)
    #train_data, test_data =load_train_test(root, train_test_path, load_txt = True)
    ##Read object exposures in each situation
    object_expo_situs = load_situs(root, situation_file)

    if train_all:
        if debug:
            train_data = minibatches['5']
        else:
            train_data = minibatches['100']

        print('Estimating the best threshold for each object in each situation ...')
        optimal_thres_situs = {}
        if  not load:
            for situ, detectors in object_expo_situs.items():
                print(' ',situ)
                gt_user_expo = gt_user_expo_situs[situ]
                optimal_thres_situs[situ] = search_optimal_thres(train_data, gt_user_expo, detectors, corr_type)
            with open(os.path.join(root, 'privacy_baseline', outdir, 'optimal_thres_situs.txt'), 'w') as fp:
                json.dump(optimal_thres_situs, fp)            
        else:
            optimal_thres_situs = json.load(open(os.path.join(root,'privacy_baseline' ,outdir, 'optimal_thres_situs.txt')))
        print('Done!')
        print('Extracting and Ranking optimal thresh per situ...')
        export_tau_ranking(optimal_thres_situs,save_file)

        print('Estimating a tau max for each situation ...')
        tau_max_situs = {}
        opt_detector_situs = {}
        opt_thresholds = {}
        for situ, gt_user_expo in gt_user_expo_situs.items():  
            print(' ',situ)

            if not cross_val:
                tau_D_max, opt_detectors, _, opt_threshold = tau_subset(train_data, gt_user_expo, optimal_thres_situs[situ], corr_type)
                tau_max_situs[situ] = tau_D_max
                opt_detector_situs[situ] = opt_detectors
                opt_thresholds[situ] = opt_threshold

            else:
                score_val_max, opt_threshold, opt_detectors = tau_max_cross_val(train_data, gt_user_expo, optimal_thres_situs[situ], corr_type, k_fold = 3)
                tau_max_situs[situ] = score_val_max
                opt_detector_situs[situ] = opt_detectors
                opt_thresholds[situ] = opt_threshold
            print(opt_detectors)

        print(tau_max_situs)
        print('Done!')
        # if plot_:
        #     plot_corr_thres_impact(corr_list_situs)
        
        print('Calculating final correlation by situation ...')
        corr_situs = {}
        for situ, gt_user_expo in gt_user_expo_situs.items():
            print(' ',situ)
            #tau_D_max = tau_max_situs[situ]
            #activated_detectors = active_subset(optimal_thres_situs[situ], tau_D_max)
            activated_detectors = opt_detector_situs[situ]
            corr_situ = corr(test_data, gt_user_expo, activated_detectors, corr_type, print_ = True, test_mode= True)
            #corr_situ = pos_neg_corr(train_data, gt_user_expo, activated_detectors, corr_type, print_=True, test_mode = False)
            corr_situs[situ] = corr_situ
        print('Done!')
        print(corr_situs)


def plot_corr_thres_impact(corr_list_situs):
    """

    :param corr_list_situs: dict
        list of correlation values computed by different groups of subsets
            {situ1: corr_list, ...}

    :return:
    """
    accronym_situs = {'job_search_IT': 'IT', 'bank_credit': 'BANK', 'job_search_waiter_waitress': 'WAIT',
                      'accommodation_search': 'ACCOM'}
    tau_fixes = np.linspace(-1, 1, 201)
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    k_fig = 0
    for situ, corr_list in corr_list_situs.items():

        corr_array = np.asarray(corr_list)
        indexes = np.where((tau_fixes >= -0.2) & (tau_fixes <= 0.2))[0]
        not_nan_corr = corr_array[indexes]
        coresp_tau_fixes = tau_fixes[indexes]

        if k_fig == 0:

            ax[0][0].plot(coresp_tau_fixes, not_nan_corr)
            ax[0][0].set_ylabel('Kendall Tau on train set')
            ax[0][0].set_xlabel('Correlation threshold for detectors')
            ax[0][0].set_title(accronym_situs[situ])
        if k_fig == 1:

            ax[0][1].plot(coresp_tau_fixes, not_nan_corr)
            ax[0][1].set_ylabel('Kendall Tau on train set')
            ax[0][1].set_xlabel('Correlation threshold for detectors')
            ax[0][1].set_title(accronym_situs[situ])
        if k_fig == 2:

            ax[1][0].plot(coresp_tau_fixes, not_nan_corr)
            ax[1][0].set_ylabel('Kendall Tau on train set')
            ax[1][0].set_xlabel('Correlation threshold for detectors')
            ax[1][0].set_title(accronym_situs[situ])
        if k_fig == 3:

            ax[1][1].plot(coresp_tau_fixes, not_nan_corr)
            ax[1][1].set_ylabel('Kendall Tau on train set')
            ax[1][1].set_xlabel('Correlation threshold for detectors')
            ax[1][1].set_title(accronym_situs[situ])

        k_fig += 1

    plt.show()
    plt.savefig('Kendall_tau_train_set.png')

if __name__ == "__main__":
    main()