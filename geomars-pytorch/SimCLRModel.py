# https://github.com/Spijkervet/SimCLR/blob/master/simclr/simclr.py
# https://github.com/leftthomas/SimCLR/blob/master/model.py
import torch.nn as nn
import torchvision
from torch import nn
import torch.nn.functional as F
import torch
import hparams as hp
from torchvision import models



def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, projection_dim):
        super(SimCLR, self).__init__()

        # densenet
        self.encoder = models.densenet121(pretrained=True)
        # self.encoder = nn.Sequential(*list(densenet_orig.children())[:-1])
        self.encoder.requires_grad_(False)
        self.n_features = hp.DENSENET_NUM_FEATURES

        # hashmodel
        self.lin1 = nn.Linear(self.n_features, 500)
        init_weights(self.lin1)
        self.leakyrelu1 = nn.LeakyReLU(negative_slope=hp.MARGIN)
        self.lin2 = nn.Linear(500, 250)
        init_weights(self.lin2)
        self.leakyrelu2 = nn.LeakyReLU(negative_slope=hp.MARGIN)
        self.lin3 = nn.Linear(250, hp.HASH_BITS)
        init_weights(self.lin3)

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(hp.HASH_BITS, hp.HASH_BITS, bias=False),
            nn.ReLU(),
            nn.Linear(hp.HASH_BITS, projection_dim, bias=False),
        )

    def forward(self, x):
        features = self.encoder(x)

        seq = nn.Sequential(self.lin1, self.leakyrelu1, self.lin2, self.leakyrelu2, self.lin3)

        h = torch.tanh(seq(features))

        z = self.projector(h)

        # Todo: -1 oder 1
        return F.normalize(z, dim=1)