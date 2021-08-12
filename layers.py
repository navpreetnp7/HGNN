import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, batch_size, in_features, out_features, agg):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size
        self.agg = agg

        weight1_eye = torch.FloatTensor(torch.eye(in_features))
        weight1_eye = weight1_eye.reshape((1, in_features, in_features))
        weight1_eye = weight1_eye.repeat(batch_size, 1, 1)
        self.weight1 = Parameter(weight1_eye)
        self.weight2 = Parameter(torch.zeros(batch_size, in_features, in_features))
        if self.agg:
            self.weight3 = Parameter(torch.zeros(batch_size, in_features, in_features))
            self.weight4 = Parameter(torch.zeros(batch_size, in_features, in_features))

    def forward(self, input, adj):
        if self.agg:
            features, aggfeatures = torch.chunk(input, chunks=2, dim=2)
            support1 = self.weight1 + torch.bmm(self.weight2, adj)
            output1 = torch.bmm(support1, features)
            support2 = self.weight3 + torch.bmm(self.weight4, adj)
            output2 = torch.bmm(support2, aggfeatures)
            return output1 + output2
        else:
            support = self.weight1 + torch.bmm(self.weight2, adj)
            output = torch.bmm(support, input)
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class InnerProduct(Module):

    def __init__(self, in_dim):
        super(InnerProduct, self).__init__()
        self.in_dim = in_dim

    def forward(self, input):
        x, y = torch.chunk(input, chunks=2, dim=2)
        y = y.permute(0, 2, 1)
        xy = torch.bmm(x, y)
        xy = torch.flatten(xy)
        return xy

    def __repr__(self):
        return self.__class__.__name__