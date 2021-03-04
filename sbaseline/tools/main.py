"""

Baseline method
    python3 main.py --config_file ../configs/mobi_BL.yaml --model_name bank_mobi1.pkl --situation BANK

"""
import os
import pickle
import argparse
import _init_paths
from sbaseline.sbaseline import SBASELINE
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

    args = argument_parser().parse_args()
    cfg = set_up(args)

    model = SBASELINE(cfg, args.situation, args.model_name)
    model.optimize()

    if cfg.OUTPUT.VERBOSE:
        print("Save modeling !!!")

    save_model(model, args.model_name, cfg.OUTPUT.DIR)
    print('Best Model Result: ', model.test_result)


if __name__ == "__main__":
    main()
