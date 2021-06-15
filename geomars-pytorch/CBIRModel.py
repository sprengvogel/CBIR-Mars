from torch import nn
import torch.nn.functional as F
import torch
import hparams as hp

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class CBIRModel(nn.Module):
    def __init__(self):
        super(CBIRModel, self).__init__()
        self.encoder = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
        self.encoder = torch.nn.Sequential(*(list(self.encoder.children())[:-1]), nn.AvgPool2d(7))
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

    def forward(self, x):
        seq = nn.Sequential(self.lin1,self.leakyrelu1,self.lin2,self.leakyrelu2,self.lin3)
        encoded_features = self.encoder(x).squeeze()
        output = seq(encoded_features)
        #print(nn.Sigmoid()(output).shape)
        #print(nn.Sigmoid()(output))
        #print(F.normalize(nn.Sigmoid()(output),1).shape)
        #print(F.normalize(nn.Sigmoid()(output),1))
        return nn.Sigmoid()(output)
