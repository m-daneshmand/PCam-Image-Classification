import time
from copy import deepcopy

import torch as th
from sklearn.neighbors import KNeighborsClassifier
from torch.utils import data

from dataloader import *
from log import print_save


class Testing:

    def __init__(self, model, data_loader, test_file='model/PCamNet'):
        self.model = model
        self.data_loader = data_loader  # List of tuple: (left_cam, right_cam, disp_map) filenames
        self.test_file = test_file

        self.model.model.eval()

    def test_model(self, type, num_neighbors=1, batch_size=128, pretrained=0):

        if type == 'knn':
            self._knn_classifier(num_neighbors, batch_size, pretrained)
        else:
            self._probability_classifier()

        return 0

    def _probability_classifier(self):
        print('Testing model:', self.model.get_name())
        acc_count = 0
        acc = 0
        softmax = th.nn.Softmax(dim=1)
        for batch_data, batch_labels in self.data_loader:
            labels = self.model.forward_pass(batch_data)

            acc += np.mean(batch_labels.numpy() == th.argmax(softmax(labels), dim=1).numpy())

            acc_count += 1

        print_save('accuracy: {}'.format(acc / acc_count))
        return 0

    def _compute_embeddings(self, pretrained, batch_size):

        data_set = PCamLoader(mode='train')
        test_X = []
        test_y = []

        for batch_data, batch_labels in self.data_loader:
            if pretrained == 1:
                test_X.append(deepcopy(self.model.compute_features(batch_data).numpy()))
                test_y.append(deepcopy(batch_labels).numpy())
            else:
                test_X.append(batch_data.numpy())
                test_y.append(batch_labels.numpy())

        test_X = np.concatenate(test_X)
        test_y = np.concatenate(test_y)

        train_dataloader = data.DataLoader(data_set, batch_size=batch_size,
                                           shuffle=False)

        train_X = []
        train_y = []

        for batch_data, batch_labels in train_dataloader:
            if pretrained == 1:
                train_X.append(self.model.compute_features(batch_data).numpy())
                train_y.append(batch_labels.numpy())
            else:
                train_X.append(batch_data.numpy())
                train_y.append(batch_labels.numpy())

        train_X = np.concatenate(train_X)
        train_y = np.concatenate(train_y)

        if pretrained == 0:
            test_X = test_X.reshape(test_X.shape[0], -1)
            train_X = train_X.reshape(train_X.shape[0], -1)

        return train_X, train_y, test_X[0:1000], test_y[0:1000]

    def _knn_classifier(self, num_neighbors, batch_size, pretrained):

        train_X, train_y, test_X, test_y = self._compute_embeddings(pretrained, batch_size)
        t1 = time.time()
        knn = KNeighborsClassifier(n_neighbors=num_neighbors)
        knn.fit(train_X, train_y)
        t2 = time.time()

        print('Fitted KNN model', 'Fit model time:', (t2 - t1))

        t3 = time.time()
        prediction = knn.predict(test_X)

        t4 = time.time()
        print('accuracy: {}'.format(np.mean(prediction == test_y)))
        print('Pred time:', (t4 - t3))
        return 0

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader
