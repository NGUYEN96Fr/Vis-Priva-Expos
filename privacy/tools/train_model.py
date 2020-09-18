"""
The module trains exposure predictors on different modeled situations

run:
    python train_model.py --config_file ../configs/svm.yaml --model_name RF1.pkl
"""
import _init_paths
import os
import argparse
import pickle
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
    parser.add_argument("--model_name", required= True, help= "saved model name")

    return parser

def save_model(model, filename, out_dir):
    root = os.getcwd()
    out_file_path = os.path.join(root, out_dir, filename)
    with open(out_file_path, 'wb') as output:  # Overwrites any existing file.
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
    args = default_argument_parser().parse_args()
    cfg = set_up(args)
    model = VISPEL(cfg)
    if cfg.OUTPUT.VERBOSE:
        print(model.cfg)

    model.train_vispel()
    if cfg.OUTPUT.VERBOSE:
        print("Save model !!!")

    save_model(model, cfg.OUTPUT.DIR, args.model_name)

if __name__ == '__main__':
    main()