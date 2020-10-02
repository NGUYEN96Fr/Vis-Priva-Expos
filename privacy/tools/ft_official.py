"""
The module fine-tunes exposure predictor on the user selection,
 and loading methods.

run:
    python ft_official.py --config_file ../configs/rf_ft.yaml --model_name best_bank.pkl --situation BANK
    python ft_official.py --config_file ../configs/rf_ft.yaml --model_name best_accom.pkl --situation ACCOM
    python ft_official.py --config_file ../configs/rf_ft.yaml --model_name best_it.pkl --situation IT
    python ft_official.py --config_file ../configs/rf_ft.yaml --model_name best_wait.pkl --situation WAIT

"""
import _init_paths
import os
import tqdm
import copy
import numpy as np
import argparse
import pickle
from situ.acronym import situ_decoding
from vispel.config import get_cfg
from vispel.vispel import VISPEL

def default_argument_parser():
    """
    Create a parser with some common arguments.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--model_name", required= True, help= "saved modeling name")
    parser.add_argument("--situation", required=True, help="IT, ACCOM, BANK, WAIT")

    return parser


def save_model(model, filename, out_dir):
    root = os.getcwd()
    out_dir_path = os.path.join(root, out_dir)
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)

    out_file_path = os.path.join(out_dir_path, filename)

    with open(out_file_path, 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)


def set_up(args):
    """

    :param args:
    :return:
        cfg
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)

    return cfg


def main():
    """

    :return:
    """
    trained_models = []
    test_corrs = []

    args = default_argument_parser().parse_args()
    cfg = set_up(args)

    EPSs = [0.05, 0.1, 0.15, 0.2]
    KEEPs = [0.8, 0.85, 0.9, 0.95, 1.0]
    DETECTOR_LOAD = [True, False]
    FEATURE_TYPEs = ['VOTE', 'ORG']
    F_TOPs = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    Ks = [10, 15, 20]
    GAMMAs = [0, 1, 2, 3, 4, 5]

    for eps in tqdm.tqdm(EPSs):
        cfg.USER_SELECTOR.EPS = eps

        for keep in KEEPs:
            cfg.USER_SELECTOR.KEEP = keep

            for load in DETECTOR_LOAD:
                cfg.DETECTOR.LOAD = load

                for feature in FEATURE_TYPEs:
                    cfg.SOLVER.FEATURE_TYPE = feature

                    for f_top in F_TOPs:
                        cfg.SOLVER.F_TOP = f_top

                        for gamma in tqdm.tqdm(GAMMAs):
                            cfg.SOLVER.GAMMA = gamma

                            for k in Ks:
                                cfg.SOLVER.K = k

                                model = VISPEL(cfg, situ_decoding(args.situation))

                                model.train_vispel()
                                trained_models.append(copy.deepcopy(model))

                                model.test_vispel()
                                test_corrs.append(model.test_result)

                                del model

    if cfg.OUTPUT.VERBOSE:
        print("Save modeling !!!")

    max_corr_index = np.argmax(test_corrs)
    best_model = trained_models[max_corr_index]
    save_model(best_model, args.model_name, cfg.OUTPUT.DIR)
    print(test_corrs)
    print('Best Model Result: ',test_corrs[max_corr_index])

if __name__ == '__main__':
    main()