"""
Train regression modeling by using
the CNN architecture

    python train_cnn.py --config_file ../configs/cnn.yaml --model_name vispel_net.pth

"""
import _init_paths
import os
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from vispel.config import get_cfg
from data.cnn_loader import data_loader, vispel_data_loader
from torch.utils.data import DataLoader
from modeling.vispel_cnn import VISPEL, pearson_loss


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

    train_dataset = vispel_data_loader(train_data)
    train_loader = DataLoader(train_dataset, batch_size=4,
                                shuffle=True, num_workers=0)

    model = VISPEL()
    criterion = nn.SmoothL1Loss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(300):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:  # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0

    print('Finished Training')
    print('Save model !!')
    root = os.getcwd()
    out_dir_path = os.path.join(root, cfg.OUTPUT.DIR)
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)
    PATH = os.path.join(out_dir_path,args.model_name)
    torch.save(model.state_dict(), PATH)