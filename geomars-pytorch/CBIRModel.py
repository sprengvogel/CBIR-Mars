from torch import nn
import torch.nn.functional as F
import torch
import hparams as hp
import torchvision
import os
from utils import load_encoder


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class CBIRModel(nn.Module):

    def __init__(self, useEncoder=True, useProjector=True):
        super(CBIRModel, self).__init__()
        self.useEncoder = useEncoder
        self.useProjector = useProjector
        
        self.encoder = load_encoder()

        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.lin1 = nn.Linear(hp.DENSENET_NUM_FEATURES, 512)
        init_weights(self.lin1)
        self.leakyrelu1 = nn.LeakyReLU(negative_slope=hp.MARGIN)
        self.lin2 = nn.Linear(512, 256)
        init_weights(self.lin2)
        self.leakyrelu2 = nn.LeakyReLU(negative_slope=hp.MARGIN)
        self.lin3 = nn.Linear(256, hp.HASH_BITS)
        init_weights(self.lin3)

        self.projector = nn.Sequential(
            nn.Linear(hp.HASH_BITS, hp.HASH_BITS, bias=False),
            nn.ReLU(),
            nn.Linear(hp.HASH_BITS, hp.PROJ_DIM, bias=False),
        )

    def forward(self, x):

        seq = nn.Sequential(self.lin1,self.leakyrelu1,self.lin2,self.leakyrelu2,self.lin3)

        if self.useEncoder:
            encoded_features = self.encoder(x).squeeze()
            output = seq(encoded_features)
        else:
            output = seq(x)

        if self.useProjector:
            z = self.projector(output)
            return nn.Sigmoid()(output), z
        else:
            return nn.Sigmoid()(output)
