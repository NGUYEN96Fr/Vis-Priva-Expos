"""
This module fine-tunes models on F_top

Use:
    python ft2_ftop.py --pre_model out/best_bank_ft1.pkl --model_name best_bank_ft2.pkl
"""


import _init_paths
import os
import tqdm
import copy
import pickle
import  argparse
import numpy as np


def default_argument_parser():
    """
    Create a parser with some common arguments.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre_model", required= True, help= "pretrained model")
    parser.add_argument("--model_name", required= True, help= "saved modeling name")

    return parser


def save_model(model, filename, out_dir):
    root = os.getcwd()
    out_dir_path = os.path.join(root, out_dir)
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)

    out_file_path = os.path.join(out_dir_path, filename)

    with open(out_file_path, 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)


def main():
    """

    :return:
    """
    args = default_argument_parser().parse_args()
    model = pickle.load(open(args.pre_model, 'rb'))

    trained_models = []
    test_corrs = []

    F_TOPs = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for f_top in tqdm.tqdm(F_TOPs):


        model.cfg.SOLVER.F_TOP = f_top
        model.set_seeds()

        model.train_vispel()
        trained_models.append(copy.deepcopy(model))

        model.test_vispel()
        test_corrs.append(model.test_result)

    if model.cfg.OUTPUT.VERBOSE:
        print("Save modeling !!!")

    max_corr_index = np.argmax(test_corrs)
    best_model = trained_models[max_corr_index]
    save_model(best_model, args.model_name, model.cfg.OUTPUT.DIR)
    print(test_corrs)
    print('Best Model Result: ',test_corrs[max_corr_index])

if __name__ == '__main__':
    main()