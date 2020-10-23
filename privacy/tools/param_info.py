"""

Usage:
    python param_info.py -p /home/nguyen/Documents/intern20/saved_models_object/it_rcnn
"""

import os
import _init_paths
import argparse
from analysis.params import best_result

def argument_parser():
    """
    Create a parser with some common arguments.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', "--path", required= True, help= "path to trained models")

    return parser

def main():
    """

    Returns
    -------
    """
    args = argument_parser().parse_args()
    # list of situations within corresponding detector
    models = os.listdir(args.path)
    best_result(args.path, models, 'FE', 'GAMMA')

if __name__ == '__main__':
    main()