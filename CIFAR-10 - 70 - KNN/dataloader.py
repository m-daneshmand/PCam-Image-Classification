import os
import pickle

import numpy as np
from torch.utils.data import Dataset


class CIFAR10Loader(Dataset):

    def __init__(self, data_path='data/CIFAR10/', mode='train', transform=None):
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
                ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
                ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
                ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
                ['data_batch_4', '634d18415352ddfa80567beed471001a'],
                ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
            ]
        elif self.mode == 'test':
            batch_list = [
                ['test_batch', '40351d587109b95175f43aff81a1287e'],
            ]
        else:
            raise Exception('Unknown: mode type(Options: train, test)')

        for batch in batch_list:
            batch = unpickle(os.path.join(self.data_path, batch[0]))
            tmp = batch[b'data']
            self.data.append(tmp)
            self.labels.append(batch[b'labels'])

        self.data = np.float32(np.concatenate(self.data))
        self.data = self.data.reshape(self.data.shape[0], 3, 32, 32)

        self.labels = np.concatenate(self.labels).astype(np.long)


def unpickle(file):
    with open(file, 'rb') as fp:
        dict = pickle.load(fp, encoding='bytes')

    return dict
