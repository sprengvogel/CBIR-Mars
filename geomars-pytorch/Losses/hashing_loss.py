from torch import nn
import torch
import hparams as hp


def hashing_criterion(anchor):
    pushLoss = nn.MSELoss(reduction='mean')

    balancing_input = [(torch.mean(x)-0.5)**2 for x in anchor.clone()]
    balancingLoss = sum(balancing_input)/len(balancing_input)

    return -hp.LAMBDA1*(1/hp.HASH_BITS)*pushLoss(anchor, 0.5*torch.ones_like(anchor)) + hp.LAMBDA2*balancingLoss

