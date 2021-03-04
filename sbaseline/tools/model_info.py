"""
This module load the trained model and reproduce its results

Use:
    python model_info.py --model_name test_/bank_mobi_0.pkl
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
    print(model.opt_threshold)
    print('#-----------------------------------#')
    print('# TEST CORR')
    print("#-----------------------------------#")
    print(model.test_result)


if __name__ == '__main__':
    info()