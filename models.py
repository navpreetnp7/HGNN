import torch.nn as nn
import torch.nn.functional as F
from layers import GraphNN1, GraphNN1_h, GraphNN2, InnerProduct

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

        x1 = self.x1(x, adj)
        h1 = self.h1(x, C)
        C_t = torch.transpose(C, 0, 1)
        C_t = C_t / C_t.sum(axis=1).unsqueeze(1)
        x2 = self.h1(x1, adj, h1, C_t)

        if self.fixed:
            mu = F.relu(self.reconstructions(x2))
            return mu, x
        else:
            lr1, lr2 = torch.chunk(x2, chunks=2, dim=2)
            mu = F.relu(self.reconstructions(lr1))
            sigma = F.relu(self.reconstructions(lr2))
            return mu, sigma, x