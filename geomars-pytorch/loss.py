from torch import nn
import torch
import hparams as hp

def criterion(anchor, pos, neg):
    tripletLoss = nn.TripletMarginLoss(margin=1)
    pushLoss = nn.MSELoss(reduction='sum')
    #print(anchor.shape)
    balancingLoss = sum([(torch.mean(x)-0.5)**2 for x in anchor.clone()])
    #print("Triplet: "+str(tripletLoss(anchor, pos, neg)))
    #print("Push: "+str(hp.LAMBDA1*(1/hp.HASH_BITS)*pushLoss(anchor, 0.5*torch.ones_like(anchor))))
    #print("balancing: "+str(hp.LAMBDA2*balancingLoss))
    return tripletLoss(anchor, pos, neg) - hp.LAMBDA1*(1/hp.HASH_BITS)*pushLoss(anchor, 0.5*torch.ones_like(anchor)) + hp.LAMBDA2*balancingLoss