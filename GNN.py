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
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
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

args = parser.parse_args(args=[])
args.cuda = not args.no_cuda and torch.cuda.is_available()

def GraphNeuralNet(adj,dim,hidden,fixed,features,C):

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
                nfeat=features.shape[1],
                hidden=hidden,
                ndim=dim,
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
            mu,lr = model(features, adj_norm, C)
            with torch.no_grad():
                mse = torch.nn.MSELoss()
                mseloss = mse(torch.flatten(mu), torch.flatten(adj))
                sig = torch.sqrt(mseloss)
            sigma = sig * torch.ones(adj.shape, requires_grad=True)
        else:
            mu, sigma,lr = model(features, adj_norm, C)

        loss = criterion(torch.flatten(adj), torch.flatten(mu), torch.flatten(torch.square(sigma)))
        loss.backward()

        optimizer.step()

        if epoch == 0:
            best_loss = loss
            best_lr = lr
            best_mu = mu
            if fixed:
                best_sig = sig
        else:
            if loss < best_loss:
                best_loss = loss
                best_lr = lr
                best_mu = mu
                if fixed:
                    best_sig = sig

        if epoch % 5000 == 0:
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss: {:.8f}'.format(best_loss.item()),
                  'time: {:.4f}s'.format(time.time() - t))

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    if fixed:
        return best_lr, best_sig, best_mu, best_loss
    else:
        return best_lr, best_mu, best_loss
