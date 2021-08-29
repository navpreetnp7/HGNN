import torch.nn as nn
import torch.nn.functional as F
from layers import GraphNN1, GraphNN1_h, GraphNN2, InnerProduct
from utils import doublerelu

import torch


class GNN(nn.Module):

    def __init__(self, batch_size, nfeat, hidden, ndim, fixed):
        super(GNN, self).__init__()

        self.batch_size = batch_size
        self.nfeat = nfeat
        self.hidden = hidden
        if fixed:
            self.ndim = 2*ndim
        else:
            self.ndim = 4*ndim
        self.fixed = fixed

        self.x1 = GraphNN1(batch_size, self.ndim, hidden, fixed)
        self.h1 = GraphNN1_h(batch_size, self.ndim, hidden, fixed)
        self.x2 = GraphNN2(batch_size, hidden, self.ndim, fixed)
        self.reconstructions = InnerProduct(self.ndim)

    def forward(self, x, adj, C):
        x = torch.sigmoid(x)
        x1 = doublerelu(self.x1(x, adj))
        C_t = torch.transpose(C, 1, 2)
        h1 = doublerelu(self.h1(x, C_t))
        C_t = C_t / C_t.sum(axis=1).unsqueeze(1)
        C_t = torch.transpose(C_t, 1, 2)
        x2 = doublerelu(self.x2(x1, adj, h1, C_t))
        x2 = torch.logit(x2)

        if self.fixed:
            mu = F.relu(self.reconstructions(x2))
            return mu, x
        else:
            lr1, lr2 = torch.chunk(x2, chunks=2, dim=2)
            mu = F.relu(self.reconstructions(lr1))
            sigma = F.relu(self.reconstructions(lr2))
            return mu, sigma, x