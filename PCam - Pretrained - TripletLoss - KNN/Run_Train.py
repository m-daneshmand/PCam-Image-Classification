import torch.backends.cudnn as cudnn
from torch.utils import data

from dataloader import *
from model import *
from train import *


def main():
    cudnn.deterministic = True

    data_set = PCamLoader(mode='train')
    net = PCamNet([32, 32, 3], 1, learning_rate=1e-4)

    data_loader = data.DataLoader(data_set, batch_size=128,
                                    shuffle=True, num_workers=0)

    net.build_model( type='siamese_pcam')
    net.load_model_encoder('model/saved_weights.pth')

    net.add_optimizer()
    train = Training(net, data_loader, batch_size=128, epochs=300)

    train.set_model_save('model/saved_weights')
    train.train_model()


if __name__ == '__main__':
    main()
