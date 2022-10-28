import pdb
import os
import time
import json
import argparse
import numpy as np
import torch
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from t_io import standard_vis as std_vis

from models import nn, nn_s_invar

DPI = 100
seed = 0
xmin = 0; xmax = 1#; ymin = -1; ymax = 1
gridsize = 50


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--system', default='System1', type=str)
    parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of mlp')
    parser.add_argument('--output_dim', default=1, type=int, help='output dimension of mlp')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--nonlinearity', default='relu', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=10000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=1000, type=int, help='number of iterations for prints')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    return parser.parse_args()

def train():
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # train loop
    stats = {'train_loss': [], 'val_loss': []}
    t0 = time.time()
    for step in range(args.total_steps+1):
        loss = model.loss(xs, target_K)
        loss.backward(); optim.step() ; optim.zero_grad()
        train_loss = loss.detach().item()

        # logging
        stats['train_loss'].append(train_loss)
        if step % args.print_every == 0:
            print("step {}, time {:.2e}, train_loss {:.4e}"
                  .format(step, time.time()-t0, train_loss))
            t0 = time.time()
    return model, stats

def vis():
    #vis kernel---------------
    fig, ax = plt.subplots(figsize=(10,10))
    for i in range(N):
        plt.plot(xs.detach().numpy(),pred_K[i].detach().numpy(),linewidth=0.5)
        plt.ylim(vmin, vmax)
    plt.title('loss=' + str(loss))
    filename = save_dir + '/kernel.pdf'
    plt.savefig(filename, format='pdf')
    plt.close()

    for i in range(N):
        filename = save_dir + '/kernel_' + str(i) + '.pdf'
        plt.plot(xs,pred_K[i].detach().numpy(),linewidth=0.5)
        plt.ylim(vmin, vmax)
        plt.savefig(filename, format='pdf')
        plt.close()

    #vis gram matrix-----------
    filename = save_dir + '/gram.pdf'
    vis_gram(filename, pred_K.detach().numpy())

    #vis learning curve
    filename = save_dir + '/learning_curve.pdf'
    x = np.arange(len(stats['train_loss']))
    std_vis.plot(filename, x, [stats['train_loss']], 'epoch','L2', ['train'])


def vis_gram(filename, data):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_aspect('equal', adjustable='box')
    sns.heatmap(data, vmin=vmin, vmax=vmax, cbar_kws={"shrink": .81},cmap='viridis')
    #sns.heatmap(data, cbar_kws={"shrink": .81},cmap='vlag')
    plt.title('loss=' + str(loss))
    plt.savefig(filename, format='pdf', dpi=DPI, bbox_inches='tight',pad_inches = 0.2)
    plt.close()
    

if __name__ == "__main__":

    args = get_args()
    xs = torch.tensor(np.linspace(xmin, xmax, gridsize))
    N = len(xs)

    filename = args.system + '/true/gram.npy'
    target_K = np.load(filename)
    target_K = torch.tensor(target_K)
    vmin = target_K.min() * 1.1 - 0.01
    vmax = target_K.max() * 1.1 + 0.01

    input_dim = 2
    #nn-----------------
    model = nn(input_dim, args.hidden_dim, args.output_dim, args.nonlinearity).double()
    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4)
    model, stats = train()
    loss = stats['train_loss'][-1]
    pred_K = model.forward(xs)
    save_dir = args.system + '/nn'
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None
    vis()
    path = '{}/model.json'.format(save_dir)
    with open(path, 'w') as f:
        json.dump(vars(args), f)

    #nn shift-invariant-----------------
    model = nn_s_invar(input_dim, args.hidden_dim, args.output_dim, args.nonlinearity).double()
    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4)
    model, stats = train()
    loss = stats['train_loss'][-1]
    pred_K = model.forward(xs)
    save_dir = args.system + '/nn_s_invar'
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None
    vis()
    path = '{}/model.json'.format(save_dir)
    with open(path, 'w') as f:
        json.dump(vars(args), f)

    pdb.set_trace()
