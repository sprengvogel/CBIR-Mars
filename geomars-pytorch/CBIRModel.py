from torch import nn
import torch
import hparams as hp



class CBIRModel(nn.Module):
    def __init__(self):
        super(CBIRModel, self).__init__()
        self.conv1 = nn.Conv2d(hp.DENSENET_NUM_FEATURES, 1024, kernel_size=1)
        self.leakyrelu1 = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=1)
        self.leakyrelu2 = nn.LeakyReLU(negative_slope=0.2)
        self.conv3 = nn.Conv2d(512, hp.HASH_BITS, kernel_size=1)

    def forward(self, x):
        x = torch.reshape(x, shape=(-1, hp.DENSENET_NUM_FEATURES, 1, 1))
        seq = nn.Sequential(self.conv1,self.leakyrelu1,self.conv2,self.leakyrelu2,self.conv3)
        output = seq(x)
        return torch.reshape(nn.Sigmoid()(output), shape=(-1, hp.HASH_BITS))
