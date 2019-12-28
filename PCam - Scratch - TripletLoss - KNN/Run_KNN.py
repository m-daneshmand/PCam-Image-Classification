from model import *
from test import *


def main():
    data_set = PCamLoader(mode='test')
    net = PCamNet([32, 32, 3], 10, learning_rate=1e-4)

    data_loader = data.DataLoader(data_set, batch_size=128,
                                  shuffle=True, num_workers=0)

    net.build_model(type='siamese_pcam')
    net.load_model_encoder('model/saved_weights.pth')

    net.load_model_encoder('model/saved_weights.pth')
    test = Testing(net, data_loader, 'model/saved_weights')
    test.test_model(type='knn', num_neighbors=5, batch_size=128, pretrained=1)


if __name__ == '__main__':
    main()
