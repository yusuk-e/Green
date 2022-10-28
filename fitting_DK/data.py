import pdb
import os
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from t_io import standard_vis as std_vis

DPI = 100
seed = 0
xmin = 0; xmax = 1#; ymin = -1; ymax = 1

gridsize = 50

def vis_gram(filename, data):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_aspect('equal', adjustable='box')
    sns.heatmap(data, vmin=vmin, vmax=vmax, cbar_kws={"shrink": .81},cmap='viridis')
    #sns.heatmap(data, cbar_kws={"shrink": .81},cmap='vlag')
    plt.savefig(filename, format='pdf', dpi=DPI, bbox_inches='tight',pad_inches = 0.2)
    plt.close()

def vis():
    
    fig, ax = plt.subplots(figsize=(10,10))
    for i in range(N):
        plt.plot(xs,K[i],linewidth=0.5)
        plt.ylim(vmin, vmax)
    filename = save_dir + '/kernel.pdf'
    plt.savefig(filename, format='pdf')
    plt.close()

    for i in range(N):
        filename = save_dir + '/kernel_' + str(i) + '.pdf'
        plt.plot(xs,K[i],linewidth=0.5)
        plt.ylim(vmin, vmax)
        plt.savefig(filename, format='pdf')
        plt.close()

    #vis gram matrix-----------
    filename = save_dir + '/gram.pdf'
    vis_gram(filename, K)

    #save----------------------
    filename = save_dir + '/gram.npy'
    np.save(filename, K)


if __name__ == "__main__":

    xs = np.linspace(xmin, xmax, gridsize)
    N = xs.size

    save_dir = 'System1/true'#両端固定の弦
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None

    #gram matrix 1-------------
    L = xmax
    K = []
    for x0 in xs:
        ks = []
        for x1 in xs:
            if x0 < x1:
                k = x0*(L-x1)/L
            else:
                k = x1*(L-x0)/L
            ks.append(k)
        K.append(ks)
    K = np.array(K)
    
    #vis kernel----------------
    vmin = K.min() * 1.1 - 0.01
    vmax = K.max() * 1.1 + 0.01
    vis()
    '''
    save_dir = 'System2/true'#Helmholtz eq with Dirichlet boundary
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None

    #gram matrix 2-------------
    h = 15 #parameter
    L = xmax
    K = []
    for x0 in xs:
        ks = []
        for x1 in xs:
            if x0 < x1:
                k = (np.sin(h*x0)*np.sin(h*(x1-1))) / (h*np.sin(h))
            else:
                k = (np.sin(h*x1)*np.sin(h*(x0-1))) / (h*np.sin(h))
            ks.append(k)
        K.append(ks)
    K = np.array(K)
    
    #vis kernel----------------
    vmin = K.min() * 1.1 - 0.01
    vmax = K.max() * 1.1 + 0.01
    vis()
    '''
    pdb.set_trace()
    
