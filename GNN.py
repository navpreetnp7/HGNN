import time
import argparse
import numpy as np

import torch
import torch.optim as optim

from utils import load_data,normalize,toy_data,norm_embed,nmi_score,svdApprox
from models import GNN


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=426, help='Random seed.')
parser.add_argument('--epochs', type=int, default=20001,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.00001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=10e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--ndim', type=int, default=2,
                    help='Embeddings dimension.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

def GraphNeuralNet(adj,dim,fixed,features,sig_fix=None):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    adj_norm = normalize(adj)

    adj = torch.FloatTensor(np.array(adj))
    adj_norm = torch.FloatTensor(np.array(adj_norm))

    # loss function
    criterion = torch.nn.GaussianNLLLoss()

    # NULL Model
    mu0 = adj.mean() * torch.ones(adj.shape[1:])
    sigma0 = adj.std() * torch.ones(adj.shape[1:])
    with torch.no_grad():
        loss0 = criterion(torch.flatten(adj), torch.flatten(mu0), torch.flatten(torch.square(sigma0)))

    # Model and optimizer

    model = GNN(batch_size=adj.shape[0],
                nfeat=adj.shape[1],
                nhid=adj.shape[1],
                ndim=args.ndim,
                mu0=adj.mean(),
                sigma0=adj.std(),
                fixed=fixed)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        adj_norm = adj_norm.cuda()

    # Train model
    t_total = time.time()

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)


    for epoch in range(args.epochs):

        t = time.time()
        model.train()
        optimizer.zero_grad()

        if fixed:
            mu,lr = model(features, adj_norm)
            with torch.no_grad():
                mse = torch.nn.MSELoss()
                mseloss = mse(torch.flatten(mu), torch.flatten(adj))
                sig = torch.sqrt(mseloss)
            sigma = sig * torch.ones(adj.shape, requires_grad=True)
        else:
            mu, sigma,lr = model(features, adj_norm)

        loss = criterion(torch.flatten(adj), torch.flatten(mu), torch.flatten(torch.square(sigma)))
        loss.backward()

        optimizer.step()

        if epoch == 0:
            best_loss = loss
            best_lr = lr
            if fixed:
                best_sig = sig
        else:
            if loss < best_loss:
                best_loss = loss
                best_lr = lr
                if fixed:
                    best_sig = sig

        if epoch % 5000 == 0:
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss: {:.8f}'.format(best_loss.item()),
                  'time: {:.4f}s'.format(time.time() - t))

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    if fixed:
        return best_lr, best_sig
    else:
        return best_lr

def GNN_embed(adj,dim,features=None):

    if features == None:
        # svd features
        svd_mu, svd_sig, svd_loss, svdembedx, svdembedy = svdApprox(adj=adj, dim=dim)
        features = torch.cat((svdembedx, svdembedy), dim=1)
        features = features.unsqueeze(dim=0)
        print("Fixed Sigma dim {}".format(dim))
        mu, loss, loss0, lr, sigma = GraphNeuralNet(adj=adj, dim=dim, fixed=True, features=features)

        sig_flex = torch.ones(lr[0].detach().shape) * torch.sqrt(sigma / dim)
        features = torch.cat((lr[0].detach(), sig_flex), dim=1)
        features = features.unsqueeze(dim=0)
        print("Flexible Sigma dim {}".format(dim))
        mu, loss, loss0, lr = GraphNeuralNet(adj=adj,dim=dim,new=True,features=features,sig_fix=sigma)
    else:
        print("Flexible Sigma dim {}".format(dim))
        mu, loss, loss0, lr = GraphNeuralNet(adj=adj,dim=dim,features=features)

    return lr
