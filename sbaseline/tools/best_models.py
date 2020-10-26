"""
The module verifies best trained models for situations within detectors

     python best_models.py --path /home/nguyen/Documents/intern20/Vis-Priva-Expos/sbaseline/tools/test_

"""

import os
import pickle
import argparse
import _init_paths


def argument_parser():
    """
    Create a parser with some common arguments.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="path to trained models")

    return parser


def main():
    """

    Returns
    -------
    """
    args = argument_parser().parse_args()
    # list of situations within corresponding detector
    models = os.listdir(args.path)

    for model in models:
        mpath = os.path.join(args.path, model)
        # loaded model
        lmodel = pickle.load(open(mpath, 'rb'))
        test_result = lmodel.test_result

        print('#-----------------------#')
        print(model.split('.pkl')[0])
        print(test_result)


if __name__ == '__main__':
    main()
