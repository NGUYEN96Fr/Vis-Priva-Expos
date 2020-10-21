"""

Baseline method
    python3 main.py --config_file ../configs/mobi_BL.yaml --model_name bank_mobi_0.pkl --situation BANK

"""
import os
import pickle
import json
import copy
import argparse
import numpy as np
import _init_paths
from baseline.baseline import BASELINE
from config.config import get_cfg


def argument_parser():
    """
    Create a parser with some common arguments.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--model_name", required=True, help="saved modeling name")
    parser.add_argument("--situation", required=True, help="IT, ACCOM, BANK, WAIT")
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    return parser


def save_model(model, filename, out_dir):
    out_dir_path = out_dir
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)

    out_file_path = os.path.join(out_dir_path, filename)

    with open(out_file_path, 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

    with open(model.save_path, 'w') as fp:
        json.dump(model.opt_thresholds, fp)


def set_up(args):
    """

    :param args:
    :return:
        cfg
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    return cfg


def main():
    """

    return
    -------

    """
    trained_models = []
    test_corrs = []

    args = argument_parser().parse_args()
    cfg = set_up(args)

    CROSS_VALs = [True]

    for cross_val in CROSS_VALs:
        cfg.SOLVER.CROSS_VAL = cross_val

        model = BASELINE(cfg, args.situation, args.model_name)
        model.optimize()

        trained_models.append(copy.deepcopy(model))
        test_corrs.append(model.test_result)
        del model

    if cfg.OUTPUT.VERBOSE:
        print("Save modeling !!!")

    max_corr_index = np.argmax(test_corrs)
    best_model = trained_models[max_corr_index]
    save_model(best_model, args.model_name, cfg.OUTPUT.DIR)
    print(test_corrs)
    print('Best Model Result: ',test_corrs[max_corr_index])

if __name__ == "__main__":
    main()
