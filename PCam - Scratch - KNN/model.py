import torch as th


# def _pickle_method(m):
#     if m.im_self is None:
#         return getattr, (m.im_class, m.im_func.func_name)
#     else:
#         return getattr, (m.im_self, m.im_func.func_name)
#
# copyreg.pickle(types.MethodType, _pickle_method)


class PCamNetVGGModel(th.nn.Module):

    def __init__(self, num_classes=2, freeze_encoder=False):
        super(PCamNetVGGModel, self).__init__()
        self.vgg_encoder = th.nn.Sequential(
            th.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            th.nn.ReLU(inplace=True),
            th.nn.Dropout(0.3),
            th.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            th.nn.ReLU(inplace=True),
            th.nn.MaxPool2d(kernel_size=2, stride=2),

            th.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            th.nn.ReLU(inplace=True),
            th.nn.Dropout(0.3),
            th.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            th.nn.ReLU(inplace=True),
            th.nn.MaxPool2d(kernel_size=2, stride=2),

            th.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            th.nn.ReLU(inplace=True),
            th.nn.Dropout(0.3),
            th.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            th.nn.ReLU(inplace=True),
            th.nn.MaxPool2d(kernel_size=2, stride=2),

            th.nn.Conv2d(256, 512, kernel_size=3, padding=1),
            th.nn.ReLU(inplace=True),
            th.nn.Dropout(0.3),
            th.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            th.nn.ReLU(inplace=True),
            th.nn.MaxPool2d(kernel_size=2, stride=2),

            th.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            th.nn.ReLU(inplace=True),
            th.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        if freeze_encoder:
            for param in self.vgg_encoder.parameters():
                param.requires_grad = False

        self.classifier = th.nn.Sequential(
            th.nn.Dropout(0.25),
            th.nn.Linear(512, 256),
            th.nn.ReLU(inplace=True),
            th.nn.Linear(256, num_classes)
        )

        self.loss = th.nn.CrossEntropyLoss()

    def forward(self, img):
        features = self.vgg_encoder(img)
        features = features.view(features.size(0), -1)
        y = self.classifier(features)
        return y

    def compute_features(self, img):
        features = self.vgg_encoder(img)
        features = features.view(features.size(0), -1)
        features = th.nn.functional.normalize(features, dim=1)
        return features

    def get_name(self):
        return 'PCamNet: a VGG like model'

    def compute_loss(self, y, y_):
        device = th.device('cuda')
        y_ = y_.type(th.LongTensor).to(device)
        celoss = self.loss(y, y_)
        return celoss


class PCamNet(object):

    def __init__(self, input_dims, num_classes, learning_rate=1e-4):
        self.input_dims = input_dims
        self.num_classes = num_classes
        self.learning_rate = learning_rate

    def build_model(self):
        self.model = PCamNetVGGModel(num_classes=self.num_classes)
        self.model = self.model.cuda()
        print(self.model.get_name(), self.model)

    def add_optimizer(self):
        self.optimizer = th.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                       lr=self.learning_rate)

    def train_batch(self, data, labels):
        data = data.cuda()
        labels = labels.cuda()

        y = self.model.forward(data)
        loss = self.model.compute_loss(y, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data.cpu()

    def compute_loss(self, data, labels):
        data = data.cuda()
        labels = labels.cuda()

        y = self.model.forward(data)
        loss = self.model.compute_loss(y, labels)

        return loss.data.cpu()

    def forward_pass(self, data):
        data = data.cuda()
        y = self.model.forward(data)
        labels = y.data

        return labels.cpu()

    def compute_features(self, data):
        data = data.cuda()
        features = self.model.compute_features(data)
        return features.detach().cpu()

    def save_model(self, ckpt_file):
        th.save(self.model.state_dict(), ckpt_file)

    def load_model(self, ckpt_file=None):
        if ckpt_file != None:
            self.model.load_state_dict(th.load(ckpt_file))
            print('loaded model:', ckpt_file)

    def load_model_encoder(self, ckpt_file):
        if ckpt_file != None:
            self.model.vgg_encoder.load_state_dict(th.load(ckpt_file), strict=False)
            print('loaded encoder ...')

    def get_name(self):
        return self.model.get_name()
