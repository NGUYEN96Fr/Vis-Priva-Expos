"""
Study feature selection impact on visual privacy prediction

usage:
    python _features.py --config_file ../configs/_features.yaml
"""

import _init_paths
import os
import argparse
import pickle
from vispel.config import get_cfg
from vispel.vispel import VISPEL

def default_argument_parser():
    """
    Create a parser with some common arguments.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required= True, metavar="FILE", help="path to config file")

    return parser

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
    print('#-----------------------------------#')
    print('# VISUAL FEATURE SELECTION')
    print("#-----------------------------------#")
    args = default_argument_parser().parse_args()
    cfg = set_up(args)

    feature_expos = {}

    features = ['ORG', 'ABS', 'POS_NEG', 'SUM']

    for feature in features:
        cfg.SOLVER.FEATURE_TYPE = feature
        model = VISPEL(cfg)
        if cfg.OUTPUT.VERBOSE:
            print(cfg.SOLVER.FEATURE_TYPE)

        model.train_vispel()
        model.test_vispel()

        for situ, corr in model.test_results.items():
            if feature not in feature_expos:
                feature_expos[feature] = []
            feature_expos[feature].append(corr)

    for feature, expos in feature_expos.items():
        feature_expos[feature].append(sum(expos)/len(expos))

    print(feature_expos)

if __name__ == '__main__':
    main()