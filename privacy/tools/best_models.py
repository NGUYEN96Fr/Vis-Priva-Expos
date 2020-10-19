"""
The module verifies best trained models for situations within detectors

     python best_models.py --path /home/nguyen/Documents/intern20/saved_models_mobi

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
    parser.add_argument("--path", required= True, help= "path to trained models")

    return parser

def main():
    """

    Returns
    -------
    """
    args = argument_parser().parse_args()
    # list of situations within corresponding detector
    sdetecs = os.listdir(args.path)

    for sdetec in sdetecs:
        # list of trained models
        spath = os.path.join(args.path, sdetec)
        models = os.listdir(spath)
        best_result = -1

        for model in models:
            mpath = os.path.join(spath, model)
            # loaded model
            lmodel = pickle.load(open(mpath, 'rb'))
            lmodel.set_seeds()
            lmodel.test_vispel()
            test_result = lmodel.test_result

            if test_result > best_result:
                best_result = test_result
                best_model = mpath
        print('#-----------------------#')
        print(sdetec)
        print(best_result)

if __name__ == '__main__':
    main()