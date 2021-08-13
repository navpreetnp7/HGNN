import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, InnerProduct
from utils import norm_embed
import torch

class GNN(nn.Module):

    def __init__(self, batch_size, nfeat, nhid, ndim, fixed, agg):
        super(GNN, self).__init__()

        self.agg = agg
        self.gc1 = GraphConvolution(batch_size, nfeat, nhid, agg)
        self.fixed = fixed
        self.embeddings = GraphConvolution(batch_size, nhid, ndim, agg)
        self.reconstructions = InnerProduct(ndim)

    def forward(self, x, adj):
        if self.agg:
            x,lr,agglr = self.gc1(x, adj)
        else:
            x = self.gc1(x, adj)
        if self.fixed:
            mu = F.relu(self.reconstructions(x))
            if self.agg:
                return mu,lr,agglr
            else:
                return mu, x
        else:
            lr1, lr2 = torch.chunk(x, chunks=2, dim=2)
            mu = F.relu(self.reconstructions(lr1))
            sigma = F.relu(self.reconstructions(lr2))
            if self.agg:
                return mu, sigma, lr, agglr
            else:
                return mu, sigma, x
