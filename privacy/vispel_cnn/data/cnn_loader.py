"""
Data loader

"""
import os
import pickle
import torch
from data.features import GData
from torch.utils.data import Dataset

def data_loader(cfg, file_name):
    """

    :param cfg:
    :return:
    """
    root = os.getcwd()
    out_dir_path = os.path.join(root, cfg.OUTPUT.DIR)
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)

    file_path = os.path.join(out_dir_path, file_name)

    if not os.path.exists(file_path):
        data_generator = GData(cfg,file_name)
        train_data, test_data = data_generator.save()

    else:
        data = pickle.load(open(file_path, 'rb'))
        train_data, test_data = data['train'], data['test']

    return train_data, test_data

class vispel_data_loader(Dataset):
    """
    Data loader for VISPEL

    :param Dataset:
    :return:

    """

    def __init__(self, data):
        self.X_features = torch.from_numpy(data['features']).float()
        self.y_expos = torch.from_numpy(data['expos']).float()


    def __len__(self):
        return self.X_features.shape[0]

    def __getitem__(self, index):

        features = self.X_features[index,:, :, :]
        expos = self.y_expos[index, :, :].view(-1)

        return features, expos


