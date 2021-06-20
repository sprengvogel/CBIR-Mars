from torch import nn
import torch
import hparams as hp
import torch.nn.functional as F


def contrastive_loss(z_i, z_j):
    # https://github.com/leftthomas/SimCLR/blob/master/main.py

    batch_size = z_i.size()[0]

    temperature = 0.5
    out = torch.cat([z_i, z_j], dim=0)
    # [2*B, 2*B]
    sim_matrix = F.cosine_similarity(out.unsqueeze(1), out.unsqueeze(0), dim=2)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()

    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(z_i * z_j, dim=-1) / temperature)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

    return loss


def criterion(z_i, z_j):
    pushLoss = nn.MSELoss(reduction='sum')
    balancingLoss_i = hp.LAMBDA2 * sum([(torch.mean(x)-0.5)**2 for x in z_i.clone()])
    balancingLoss_j = hp.LAMBDA2 * sum([(torch.mean(x) - 0.5) ** 2 for x in z_j.clone()])
    pushLoss_i = hp.LAMBDA1*(1/hp.HASH_BITS)*pushLoss(z_i, 0.5*torch.ones_like(z_i))
    pushLoss_j = hp.LAMBDA1 * (1 / hp.HASH_BITS) * pushLoss(z_j, 0.5 * torch.ones_like(z_j))

    return contrastive_loss(z_i, z_j) - ((pushLoss_i + pushLoss_j) / 2) + ((balancingLoss_i + balancingLoss_j) / 2)