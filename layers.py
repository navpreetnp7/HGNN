import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):

    def __init__(self, batch_size, in_features, ndim, agg, fixed):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.batch_size = batch_size
        self.agg = agg
        self.fixed = fixed
        self.ndim = ndim

        weight1_eye = torch.FloatTensor(torch.eye(in_features))
        weight1_eye = weight1_eye.reshape((1, in_features, in_features))
        weight1_eye = weight1_eye.repeat(batch_size, 1, 1)
        self.weight1 = Parameter(weight1_eye)
        self.weight2 = Parameter(torch.zeros(batch_size, in_features, in_features))
        self.weight3 = Parameter(torch.zeros(batch_size, in_features, in_features))
        self.weight4 = Parameter(torch.zeros(batch_size, in_features, in_features))


    def forward(self, input, adj):
        if self.agg:
            if self.fixed:
                features, aggfeatures = torch.split(input, split_size_or_sections=[2*self.ndim,input.shape[2]-2*self.ndim], dim=2)
            else:
                features, aggfeatures = torch.split(input, split_size_or_sections=[4*self.ndim,input.shape[2]-4*self.ndim], dim=2)
            support1 = self.weight1 + torch.bmm(self.weight2, adj)
            output1 = torch.bmm(support1, features)
            support2 = self.weight3 + torch.bmm(self.weight4, adj)
            output2 = torch.bmm(support2, aggfeatures)
            if self.fixed:
                nb_aggfeat = int(output2.shape[2]/2)
                output1l,output1r = torch.chunk(output1,chunks=2,dim=2)
                output2l, output2r = torch.chunk(output2, chunks=2, dim=2)
                output1l[:,:,:nb_aggfeat] = output1l[:,:,:nb_aggfeat] + output2l
                output1r[:, :, :nb_aggfeat] = output1r[:, :, :nb_aggfeat] + output2r
            else:
                nb_aggfeat = int(output2.shape[2]/4)
                output1m, output1s = torch.chunk(output1, chunks=2, dim=2)
                output1ml, output1mr = torch.chunk(output1m, chunks=2, dim=2)
                output1sl, output1sr = torch.chunk(output1s, chunks=2, dim=2)
                output2m, output2s = torch.chunk(output2, chunks=2, dim=2)
                output2ml, output2mr = torch.chunk(output2m, chunks=2, dim=2)
                output2sl, output2sr = torch.chunk(output2s, chunks=2, dim=2)
                output1ml[:, :, :nb_aggfeat] = output1ml[:, :, :nb_aggfeat] + output2ml
                output1sl[:, :, :nb_aggfeat] = output1sl[:, :, :nb_aggfeat] + output2sl
                output1mr[:, :, :nb_aggfeat] = output1mr[:, :, :nb_aggfeat] + output2mr
                output1sr[:, :, :nb_aggfeat] = output1sr[:, :, :nb_aggfeat] + output2sr

            return output1,output2
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