"""
This module load the trained modeling and reproduce its results

Use:
    python model_info.py --model_name out/RF1.pkl
"""

import pickle
import _init_paths
import  argparse


def default_argument_parser():
    """
    Create a parser with some common arguments.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required= True, help= "saved modeling name")

    return parser


def main():
    """

    :return:
    """
    args = default_argument_parser().parse_args()
    model = pickle.load(open(args.model_name,'rb'))
    print('#-----------------------------------#')
    print('# MODEL CONFIGURATION')
    print("#-----------------------------------#")
    print(model.cfg)
    print('#-----------------------------------#')
    print('# MODEL TRAINING RESULTS')
    print("#-----------------------------------#")
    model.train_vispel()
    # Print the modeling test results
    model.test_vispel()


if __name__ == '__main__':
    main()