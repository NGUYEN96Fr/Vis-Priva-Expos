"""

Test CNN architecture

    python test_cnn.py --config_file ../configs/cnn.yaml --model_name vispel_net.pth

"""
import _init_paths
import os
import numpy as np
import argparse
import torch
from vispel.config import get_cfg
from data.cnn_loader import data_loader, vispel_data_loader
from torch.utils.data import DataLoader
from modeling.vispel_cnn import VISPEL
from corr.corr_type import pear_corr, kendall_corr

def default_argument_parser():
    """
    Create a parser with some common arguments.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required= True, metavar="FILE", help="path to config file")
    parser.add_argument("--model_name", required= True, help= "saved modeling name")
    parser.add_argument("--data_file", default="cnn_data.pkl", metavar="FILE", help="path to config file")

    return parser

def set_up(args):
    """

    :param args:
    :return:
        cfg
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    return cfg


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    cfg = set_up(args)
    train_data, test_data = data_loader(cfg, args.data_file)

    test_dataset = vispel_data_loader(test_data)
    test_loader = DataLoader(test_dataset, batch_size=1,
                                shuffle=False, num_workers=0)


    model = VISPEL()
    root = os.getcwd()
    out_dir_path = os.path.join(root, cfg.OUTPUT.DIR)
    PATH = os.path.join(out_dir_path,args.model_name)
    model.load_state_dict(torch.load(PATH))

    situ1_gt = []
    situ1_pred =[]
    situ2_gt = []
    situ2_pred =[]
    situ3_gt = []
    situ3_pred =[]
    situ4_gt = []
    situ4_pred =[]

    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        outputs = model(inputs).data.numpy()

        situ1_gt.append(labels[0,0])
        situ1_pred.append(outputs[0,0])

        situ2_gt.append(labels[0,1])
        situ2_pred.append(outputs[0,1])

        situ3_gt.append(labels[0,2])
        situ3_pred.append(outputs[0,2])

        situ4_gt.append(labels[0,3])
        situ4_pred.append(outputs[0,3])

    situ1_gt = np.asarray(situ1_gt)
    situ1_pred = np.asarray(situ1_pred)
    situ2_gt = np.asarray(situ2_gt)
    situ2_pred = np.asarray(situ2_pred)
    situ3_gt = np.asarray(situ3_gt)
    situ3_pred = np.asarray(situ3_pred)
    situ4_gt = np.asarray(situ4_gt)
    situ4_pred = np.asarray(situ4_pred)

    if cfg.SOLVER.CORR_TYPE == 'PEARSON':
        print('situ1: ', pear_corr(situ1_gt, situ1_pred))
        print('situ2: ', pear_corr(situ2_gt, situ2_pred))
        print('situ3: ', pear_corr(situ3_gt, situ3_pred))
        print('situ4: ', pear_corr(situ4_gt, situ4_pred))