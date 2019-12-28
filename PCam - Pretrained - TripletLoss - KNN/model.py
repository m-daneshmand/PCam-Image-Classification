import numpy as np

import torch as th
import torch.utils.model_zoo as model_zoo

import copyreg
import types
import multiprocessing


class PCamNetVGGModel(th.nn.Module):

    def __init__(self, num_classes=10, freeze_encoder = False):
        super(PCamNetVGGModel, self).__init__()
        self.sigmoid = th.nn.Sigmoid()
        self.vgg_encoder = th.nn.Sequential(
                            th.nn.Conv2d(3, 64, kernel_size=3, padding=1),
                            th.nn.ReLU(inplace=True),
                            th.nn.Dropout(0.3),
                            th.nn.Conv2d(64, 64, kernel_size=3, padding=1),
                            th.nn.ReLU(inplace=True),
                            th.nn.MaxPool2d(kernel_size=2, stride=2),
                            #th.nn.BatchNorm2d(64),

                            th.nn.Conv2d(64, 128, kernel_size=3, padding=1),
                            th.nn.ReLU(inplace=True),
                            th.nn.Dropout(0.3),
                            th.nn.Conv2d(128, 128, kernel_size=3, padding=1),
                            th.nn.ReLU(inplace=True),
                            th.nn.MaxPool2d(kernel_size=2, stride=2),
                            #th.nn.BatchNorm2d(128),

                            th.nn.Conv2d(128, 256, kernel_size=3, padding=1),
                            th.nn.ReLU(inplace=True),
                            th.nn.Dropout(0.3),
                            th.nn.Conv2d(256, 256, kernel_size=3, padding=1),
                            th.nn.ReLU(inplace=True),
                            th.nn.MaxPool2d(kernel_size=2, stride=2),
                            #th.nn.BatchNorm2d(256),

                            th.nn.Conv2d(256, 512, kernel_size=3, padding=1),
                            th.nn.ReLU(inplace=True),
                            th.nn.Dropout(0.3),
                            th.nn.Conv2d(512, 512, kernel_size=3, padding=1),
                            th.nn.ReLU(inplace=True),
                            th.nn.MaxPool2d(kernel_size=2, stride=2),
                            #th.nn.BatchNorm2d(512),

                            th.nn.Conv2d(512, 512, kernel_size=3, padding=1),
                            th.nn.ReLU(inplace=True),
                            th.nn.MaxPool2d(kernel_size=2, stride=2),
                        )



        self.classifier = th.nn.Sequential(
                            th.nn.Dropout(0.25),  # classifier:add(nn.Dropout(0.5))
                            th.nn.Linear(512,256),
                            #th.nn.BatchNorm1d(256),  # classifier:add(nn.BatchNormalization(512))
                            th.nn.ReLU(inplace=True),
                            th.nn.Linear(256, num_classes)
                        )

        if freeze_encoder:
            for name, param in self.named_parameters():
                if name in ['classifier.3.weight', 'classifier.3.bias','classifier.1.weight','classifier.1.bias', 'vgg_encoder.24.weight','vgg_encoder.24.bias']:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        self.loss = th.nn.BCELoss()

    def forward(self, img):
        features = self.vgg_encoder(img)
        features = features.view(features.size(0), -1)
        y = self.classifier(features)
        y = self.sigmoid(y)
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
        y_=y_.type(th.FloatTensor).to(device)
        celoss = self.loss(y, y_)
        return celoss


class PCamNetVGGSiameseModel(PCamNetVGGModel):

    def __init__(self, num_classes=2, freeze_encoder = False):
        super(PCamNetVGGSiameseModel, self).__init__(num_classes, freeze_encoder)
        self.sigmoid = th.nn.Sigmoid()

    def compute_features(self, img):
        features = self.vgg_encoder(img)
        features = self.sigmoid(features)
        features = features.view(features.size(0), -1)
        features = th.nn.functional.normalize(features, dim=1)
        return features

    def forward(self, img):
        embeddings = self.vgg_encoder(img)
        embeddings = self.sigmoid(embeddings)
        embeddings = embeddings.view(embeddings.size(0), -1)
        embeddings = th.nn.functional.normalize(embeddings, dim=1)
        return embeddings

    def get_name(self):
        return 'PCamNet: Siamese net with VGG like encoder'

    def compute_loss(self, y, y_):
        celoss = self._compute_batch_all_triplet_loss(y, y_, 0.2)
        # celoss = self._compute_batch_hard_triplet_loss(y, y_, 0.2)

        return celoss

    def _compute_batch_all_triplet_loss(self, embeddings, y_, margin):
        # Get the pairwise distance matrix
        pairwise_dist = self._compute_pairwise_distances(embeddings)

        anchor_pos_dist = pairwise_dist.unsqueeze(2)
        anchor_neg_dist = pairwise_dist.unsqueeze(1)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        triplet_loss = anchor_pos_dist - anchor_neg_dist + margin
        all_triplet_mask = self._get_all_triplet(y_)

        triplet_loss = th.mul(all_triplet_mask, triplet_loss)
        # Check to ensure all distances >= 0
        triplet_loss = th.max(triplet_loss, th.Tensor([0.0]).cuda())

        valid_triplets = th.ge(triplet_loss, 1e-16).type(th.cuda.FloatTensor)
        num_pos_triplets = th.sum(valid_triplets)
        num_valid_triplets = th.sum(all_triplet_mask)

        fraction_pos_triplets = num_pos_triplets / (num_valid_triplets + th.Tensor([1e-16]).cuda())
        # Compute mean triplet loss over all positive valid triplets

        triplet_loss = th.sum(triplet_loss) / (num_pos_triplets + th.Tensor([1e-16]).cuda())


        return triplet_loss

    def _compute_batch_hard_triplet_loss(self, embeddings, y_, margin):
        pairwise_dist = self._compute_pairwise_distances(embeddings)
        anchor_pos_mask = self._get_anchor_positive_triplet(y_).type(th.cuda.FloatTensor)

        anchor_pos_dist = th.mul(anchor_pos_mask, pairwise_dist)

        anchor_hard_pos_dist, _ = th.max(anchor_pos_dist, dim=1, keepdim=True)

        anchor_neg_mask = self._get_anchor_negative_triplet(y_).type(th.cuda.FloatTensor)

        max_anchor_neg_dist, _ = th.max(pairwise_dist, dim=1, keepdim=True)

        anchor_neg_dist = pairwise_dist + th.mul(max_anchor_neg_dist, th.eye(1).cuda() - anchor_neg_mask)

        anchor_hard_neg_dist, _ = th.min(anchor_neg_dist, dim=1, keepdim=True)


        triplet_loss = anchor_hard_pos_dist - anchor_hard_neg_dist + margin

        # Check to ensure all distances >= 0
        triplet_loss = th.max(triplet_loss, th.Tensor([0.0]).cuda())

        # Compute mean triplet loss over hard triplets
        triplet_loss = th.mean(triplet_loss)

        return triplet_loss


    def _compute_pairwise_distances(self, embeddings, squared=False):
        # a * b.T
        dot_product = th.matmul(embeddings, th.t(embeddings))
        embed_squares = th.diag(dot_product)
        # pairwise distances
        # ||a - b||^2 = ||a||^2 - 2a * b.T  + ||b||^2
        square_distances = embed_squares.view(-1,1) - 2.0 * dot_product + embed_squares.view(1,-1)
        # Make sure the distances are >= 0
        square_distances = th.max(square_distances, th.Tensor([0.0]).cuda())

        if not squared:
            mask = th.eq(square_distances, 0.0).type(th.cuda.FloatTensor)
            square_distances = square_distances + mask * th.Tensor([1e-16]).cuda()

            square_distances = th.sqrt(square_distances)
            square_distances = th.mul(square_distances, 1.0 - mask)

        return square_distances

    def _get_anchor_positive_triplet(self, labels):
        """Returns a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
        """
        # Check pairwise labels are eq, i.e, label[i] == label[j]
        labels_eq = th.eq(labels.view(-1,1), labels.view(1,-1)).type(th.FloatTensor)
        # Compute all indices pairs (i,j), where i != j
        indices_not_eq = 1 - th.eye(labels.shape[0])

        pos_mask =th.mul(labels_eq, indices_not_eq)
        return pos_mask


    def _get_anchor_negative_triplet(self, labels):
        """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
        """
        # Check if label[i] != label[j]
        labels_eq = th.eq(labels.view(-1,1), labels.view(1,-1))

        neg_mask = 1 - labels_eq
        return neg_mask

    def _get_all_triplet(self, labels):
        """Return a 2D mask where mask[a, p, n] is True iff (a, p, n) is a valid triplet
        """
        indices_not_eq = th.Tensor([1]).cuda() - th.eye(labels.shape[0]).cuda()
        i_not_eq_j = indices_not_eq.unsqueeze(2)
        i_not_eq_k = indices_not_eq.unsqueeze(1)
        j_not_eq_k = indices_not_eq.unsqueeze(0)

        distinct_indices = th.mul(th.mul(i_not_eq_j, i_not_eq_k), j_not_eq_k)
        labels_eq = th.eq(labels.view(-1,1).cuda(), labels.view(1,-1).cuda()).type(th.cuda.FloatTensor)

        i_eq_j = labels_eq.unsqueeze(2)
        i_eq_k = labels_eq.unsqueeze(1)

        valid_labels = th.mul(i_eq_j, th.Tensor([1]).cuda() - i_eq_k)
        all_triplet_mask = th.mul(distinct_indices, valid_labels)

        return all_triplet_mask

class PCamNet(object):

    def __init__(self, input_dims, num_classes, learning_rate = 1e-4):
        self.input_dims = input_dims
        self.num_classes = num_classes
        self.learning_rate = learning_rate

    def build_model(self, type='siamese_pcam'):
        if type == 'pcam':
            self.model = PCamNetVGGModel(num_classes=self.num_classes, freeze_encoder=True)
        elif type == 'siamese_pcam':
            self.model = PCamNetVGGSiameseModel(num_classes=self.num_classes)
        else:
            raise Exception('Unknown: model type(Options: pcam, siamese_pcam)')

        self.model = self.model.cuda()
        print (self.model.get_name(), self.model)
        print('/////////////////////////////////////')
        for name, param in self.model.named_parameters():
            print(name, ':', param.requires_grad)

    def add_optimizer(self):
        self.optimizer = th.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = self.learning_rate)


    def train_batch(self, data, labels):
        data = data.cuda()
        labels = labels.cuda()

        y = self.model.forward( data )
        loss = self.model.compute_loss( y, labels )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data.cpu()

    def compute_loss(self, data, labels):
        data = data.cuda()
        labels = labels.cuda()

        y = self.model.forward( data )
        loss = self.model.compute_loss( y, labels )

        return loss.data.cpu()

    def forward_pass(self, data):
        data = data.cuda()
        y = self.model.forward( data )
        labels = y.data

        return labels.cpu()

    def compute_features(self, data):
        data = data.cuda()
        features = self.model.compute_features( data )
        return features.detach().cpu()

    def save_model(self, ckpt_file):
        th.save(self.model.state_dict(), ckpt_file)

    def load_model(self, ckpt_file = None):
        if ckpt_file != None:
            '''
            temp=PCamNetVGGModel(num_classes=10, freeze_encoder=True)
            model=th.load(ckpt_file)
            temp.load_state_dict(model)
            #temp.classifier  = th.nn.Linear(256, 1)
            temp.classifier[3] = th.nn.Linear(256, 1)
            print ("HHHHHHH", (temp),"HHHHHHH")
            #self.model = model.classifier.in_features
            #model.classifier = th.nn.Linear(10, 1)
            #self.model.classifier.__dict__[4] = th.nn.Linear(10, 1)
            #self.model.classifier.add_module(th.nn.Linear(10, 50))
            device = th.device('cuda')
            self.model=temp.to(device)#.load_state_dict(temp)
            #self.model.eval()
            print('/////////////////////////////////////')
            for name, param in self.model.named_parameters():
                print(name, ':', param.requires_grad)
            print('loaded model:', ckpt_file)
            '''
            self.model.load_state_dict(th.load(ckpt_file))
            print('loaded model:', ckpt_file)

    def load_model_encoder(self, ckpt_file):
        if ckpt_file != None:
            self.model.vgg_encoder.load_state_dict(th.load(ckpt_file), strict=False)
            print('loaded encoder ...')

    def get_name(self):
        return self.model.get_name()
