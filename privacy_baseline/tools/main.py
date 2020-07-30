import os
import sys
import configparser
import _init_paths
import json
from optimal_search.max_tau_subset import tau_subset
from optimal_search.optimal_subset import active_subset
from optimal_search.optimal_thres_object import search_optimal_thres
from optimal_search.correlation import corr
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
    siutation_file = conf['situation_path']
    outdir = conf['outdir']
    if not os.path.exists(os.path.join(root, 'privacy_baseline', outdir)):
        os.makedirs(os.path.join(root, 'privacy_baseline', outdir))
    
    corr_type = 'kendall_corr'
    debug = False
    load = True

    ##Load crowdsourcing user privacy exposure scores in each situation
    gt_user_expo_situs = load_gt_user_expo(root, gt_expo_path)
    ##Load train and test data
    minibatches, test_data = load_train_test(root, train_test_path)
    ##Read object exposures in each situation
    object_expo_situs = load_situs(root, siutation_file)

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
            with open(os.path.join(root, 'privacy_baseline', outdir, 'optimal_thres_situs.json'), 'w') as fp:
                json.dump(optimal_thres_situs, fp)            
        else:
            optimal_thres_situs = json.load(open(os.path.join(root,'privacy_baseline' ,outdir, 'optimal_thres_situs.json')))
        print('Done!')

        print('Estimating a taux max for each situation ...')
        tau_max_situs = {}
        for situ, gt_user_expo in gt_user_expo_situs.items():  
            print(' ',situ)
            tau_max_situs[situ] = tau_subset(train_data, gt_user_expo, optimal_thres_situs[situ], corr_type)
        print('Done!')
        
        print('Calculating final correlation by situation ...')
        corr_situs = {}
        for situ, gt_user_expo in gt_user_expo_situs.items():
            print(' ',situ)
            tau_D_max = tau_max_situs[situ]
            activated_detectors = active_subset(optimal_thres_situs[situ], tau_D_max)
            corr_situ = corr(train_data, gt_user_expo, activated_detectors, corr_type, infer_mode = True, print_ = True)
            corr_situs[situ] = corr_situ
        print('Done!')
        print(corr_situs)

if __name__ == "__main__":
    main()