import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, InnerProduct
from utils import norm_embed
import torch


class GNN(nn.Module):

    def __init__(self, batch_size, nfeat, nhid, ndim, mu0, sigma0, fixed):
        super(GNN, self).__init__()

        self.gc1 = GraphConvolution(batch_size, nfeat, nhid)
        self.fixed = fixed
        self.embeddings = GraphConvolution(batch_size, nhid, ndim)
        self.reconstructions = InnerProduct(ndim)

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        if self.fixed:
            mu = F.relu(self.reconstructions(x))
            return mu, x
        else:
            lr1, lr2 = torch.chunk(x, chunks=2, dim=2)
            mu = F.relu(self.reconstructions(lr1))
            sigma = F.relu(self.reconstructions(lr2))
            return mu, sigma, x
