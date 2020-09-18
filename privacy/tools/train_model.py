"""
The module trains exposure predictors on different modeled situations

run:
    python train_model.py --config_file ../configs/svm.yaml --out_dir out
"""
import _init_paths
import os
import argparse
from vispel.config import get_cfg
from trainer.vispel_trainer import VISPEL


def default_argument_parser():
    """
    Create a parser with some common arguments.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--out_dir", default="out", help="common output dir")

    return parser

def set_up(args):
    """

    :param args:
    :return:
        cfg
    """
    out_root = args.out_dir
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.OUTPUT.DIR = os.path.join(out_root, cfg.OUTPUT.DIR)
    if not os.path.exists(cfg.OUTPUT.DIR):
        os.makedirs(cfg.OUTPUT.DIR)

    return cfg


def main():
    """

    :return:
    """
    args = default_argument_parser().parse_args()
    cfg = set_up(args)
    model = VISPEL(cfg)
    if cfg.OUTPUT.VERBOSE:
        print(model.cfg)

    model.train_vispel()

if __name__ == '__main__':
    main()