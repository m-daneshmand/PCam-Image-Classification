import os
import pickle

import h5py
import numpy as np
from torch.utils.data import Dataset


def unpickle(file):
    with open(file, 'rb') as fp:
        dict = pickle.load(fp, encoding='bytes')

    return dict


class PCamLoader(Dataset):

    def __init__(self, data_path='data/', mode='train', transform=None):
        self.data_path = data_path
        self.transform = transform
        self.mode = mode

        self.data = []
        self.labels = []
        self._init_loader()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample_train = self.data[idx]
        label_train = self.labels[idx]

        if self.transform:
            sample_train = self.transform(sample_train)
            label_train = self.transform(label_train)

        return sample_train, label_train

    def _init_loader(self):
        if self.mode == 'train':
            batch_list = [
                'camelyonpatch_level_2_split_train_x.h5',
                'camelyonpatch_level_2_split_train_y.h5'
            ]
        elif self.mode == 'valid':
            batch_list = [
                'camelyonpatch_level_2_split_valid_x.h5',
                'camelyonpatch_level_2_split_valid_y.h5'
            ]
        else:
            batch_list = [
                'camelyonpatch_level_2_split_test_x.h5',
                'camelyonpatch_level_2_split_test_y.h5'
            ]

        data_file, label_file = batch_list

        self.data = np.array(extract_hdf5(
            os.path.join(self.data_path, data_file)
        )['x'][:, 32:64, 32:64, :],
                             dtype=np.float32).swapaxes(1, 2).swapaxes(1, 3)

        self.labels = np.array(
            extract_hdf5(os.path.join(self.data_path, label_file))['y'],
        ).astype(np.long).reshape(-1)

        return 0


def extract_hdf5(filename):
    f = h5py.File(filename, 'r')
    return f
