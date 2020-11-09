"""
The module verifies best trained models for situations within detectors

     python best_models.py --path /home/nguyen/Documents/intern20/models_FEo_pooling_VISPEL
     python best_models.py --path /home/nguyen/Documents/intern20/models_POOLINGx2_VISPEL
     python best_models.py --path /home/nguyen/Documents/intern20/models_POOLING_VISPEL
     python best_models.py --path /home/nguyen/Documents/intern20/models_OBJECT_VISPEL
     python best_models.py --path /home/nguyen/Documents/intern20/models_ORG
     python best_models.py --path /home/nguyen/Documents/intern20/models_OBJECT_VISPEL_31_10

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
    mobis = []
    rcnns = []

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
                best_path = mpath
        print('#-----------------------#')
        print(sdetec)
        if 'rcnn' in sdetec:
            rcnns.append(best_result)

        if 'mobi' in sdetec:
            mobis.append(best_result)

        print(best_result)
        # best_model = pickle.load(open(best_path, 'rb'))
        # print(best_model.cfg)

    print('RCNN avg: ',sum(rcnns)/len(rcnns))
    print('MOBI avg: ',sum(mobis)/len(mobis))

if __name__ == '__main__':
    main()