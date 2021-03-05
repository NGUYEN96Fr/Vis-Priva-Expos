"""
This module load the trained model and reproduce its results

Use:
    python model_info.py --model_name /home/nguyen/Documents/intern20/models_baseline_FE/bank_mobi/bank_mobi_10.pkl
"""

import pickle
import _init_paths
import  argparse


def argument_parser():
    """
    Create a parser with some common arguments.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required= True, help= "saved modeling name")

    return parser


def info():
    """

    :return:
    """
    args = argument_parser().parse_args()
    model = pickle.load(open(args.model_name,'rb'))
    model.cfg.OUTPUT.VERBOSE = True
    print('#-----------------------------------#')
    print('# MODEL CONFIGURATION')
    print("#-----------------------------------#")
    print(model.cfg)
    print('#-----------------------------------#')
    print('# TEST CORR')
    print("#-----------------------------------#")
    print(model.opt_detectors)


if __name__ == '__main__':
    info()