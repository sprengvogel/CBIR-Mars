from torch import nn
import torch
import hparams as hp

def criterion(anchor):
    pushLoss = nn.MSELoss(reduction='mean')
    #print(anchor.shape)
    balancing_input = [(torch.mean(x)-0.5)**2 for x in anchor.clone()]
    balancingLoss = sum(balancing_input)/len(balancing_input)
    #print("Triplet: "+str(tripletLoss(anchor, pos, neg)))
    #print("Push: "+str(hp.LAMBDA1*(1/hp.HASH_BITS)*pushLoss(anchor, 0.5*torch.ones_like(anchor))))
    #print("balancing: "+str(hp.LAMBDA2*balancingLoss))
    #print("push loss: ", -hp.LAMBDA1*(1/hp.HASH_BITS)*pushLoss(anchor, 0.5*torch.ones_like(anchor)))
    #print("balancing loss: ", hp.LAMBDA2*balancingLoss)
    return -hp.LAMBDA1*(1/hp.HASH_BITS)*pushLoss(anchor, 0.5*torch.ones_like(anchor)) + hp.LAMBDA2*balancingLoss
    #return tripletLoss(anchor, pos, neg) - hp.LAMBDA1*(1/hp.HASH_BITS)*pushLoss(anchor, 0.5*torch.ones_like(anchor)) + hp.LAMBDA2*balancingLoss
