"""
Study feature selection impact on visual privacy prediction

usage:
    python _focal_expo.py --config_file ../configs/_focal_expo.yaml

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
    print('# FOCAL EXPOSURE')
    print("#-----------------------------------#")
    args = default_argument_parser().parse_args()
    cfg = set_up(args)

    gamma_expos = {}

    gammas = [0, 1, 2, 3, 4, 5]

    for gamma in gammas:
        cfg.SOLVER.GAMMA = gamma
        model = VISPEL(cfg)
        if cfg.OUTPUT.VERBOSE:
            print(cfg.SOLVER.GAMMA)

        model.train_vispel()
        model.test_vispel()

        for situ, corr in model.test_results.items():
            if gamma not in gamma_expos:
                gamma_expos[gamma] = []
            gamma_expos[gamma].append(corr)

    for gamma, expos in gamma_expos.items():
        gamma_expos[gamma].append(sum(expos)/len(expos))

    print(gamma_expos)

if __name__ == '__main__':
    main()