import torch.backends.cudnn as cudnn

from model import PCamNet
from test import *
from train import *


def main():
    cudnn.deterministic = True

    data_set = PCamLoader(mode='train')
    net = PCamNet([32, 32, 3], 10, learning_rate=1e-4)

    data_loader = data.DataLoader(data_set, batch_size=128, shuffle=True, num_workers=0)

    net.build_model()

    net.add_optimizer()
    train = Training(net, data_loader,
                     batch_size=128,
                     epochs=70)

    train.set_model_save('model/saved_weights')

    train.train_model()


if __name__ == '__main__':
    main()
