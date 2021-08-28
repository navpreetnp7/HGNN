import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphNN1(Module):

    def __init__(self, batch_size, in_features, out_features, fixed):
        super(GraphNN1, self).__init__()

        self.batch_size = batch_size
        self.in_features = in_features
        self.out_features = out_features
        self.fixed = fixed

        weight0_eye = torch.FloatTensor(torch.eye(in_features,out_features))
        weight0_eye = weight0_eye.reshape((1, in_features, out_features))
        weight0_eye = weight0_eye.repeat(batch_size, 1, 1)
        self.weight0 = Parameter(weight0_eye)
        self.weight1 = Parameter(torch.zeros(batch_size, in_features, out_features))


    def forward(self, input, adj):
        print(input.shape)
        print(self.weight0.shape)
        v1 = torch.bmm(input, self.weight0)
        v2 = torch.bmm(torch.bmm(adj, input),self.weight1)
        output = v1 + v2
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphNN1_h(Module):

    def __init__(self, batch_size, in_features, out_features, fixed):
        super(GraphNN1_h, self).__init__()

        self.batch_size = batch_size
        self.in_features = in_features
        self.out_features = out_features
        self.fixed = fixed

        self.weight2 = Parameter(torch.zeros(batch_size, in_features, out_features))


    def forward(self, input, C):
        output = torch.bmm(torch.bmm(C, input),self.weight2)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphNN2(Module):

    def __init__(self, batch_size, in_features, out_features, fixed):
        super(GraphNN2, self).__init__()

        self.batch_size = batch_size
        self.in_features = in_features
        self.out_features = out_features
        self.fixed = fixed

        weight3_eye = torch.FloatTensor(torch.eye(in_features,out_features))
        weight3_eye = weight3_eye.reshape((1, in_features, out_features))
        weight3_eye = weight3_eye.repeat(batch_size, 1, 1)
        self.weight3 = Parameter(weight3_eye)
        self.weight4 = Parameter(torch.zeros(batch_size, in_features, out_features))
        self.weight5 = Parameter(torch.zeros(batch_size, in_features, out_features))


    def forward(self, input, adj, h1, C_t):
        v1 = torch.bmm(input, self.weight3)
        v2 = torch.bmm(torch.bmm(adj, input),self.weight4)
        v3 = torch.bmm(torch.bmm(C_t, h1),self.weight5)
        output = v1 + v2 + v3
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