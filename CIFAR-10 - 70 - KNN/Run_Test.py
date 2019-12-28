import torch.backends.cudnn as cudnn

from model import PCamNet
from test import *


def main():
    cudnn.deterministic = True

    data_set = CIFAR10Loader(mode='test')
    net = PCamNet([32, 32, 3], 10, learning_rate=1e-4)

    data_loader = data.DataLoader(data_set, batch_size=128, shuffle=True, num_workers=0)

    net.build_model()

    net.load_model_encoder('model/saved_weights.pth')
    test = Testing(net, data_loader, 'model/saved_weights')
    test.test_model('notknn')


if __name__ == '__main__':
    main()
