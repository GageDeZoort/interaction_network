import os
import random
import time
import argparse

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt

from models.interaction_network import InteractionNetwork
from models.graph import Graph, save_graphs, load_graph

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('train_IN.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/train_LP.yaml')
    return parser.parse_args()

def get_graphs(d):
    """ get all graphs from a directory
          return [("event<id1>", graph1, size), ...]
    """
    files = os.listdir(d)
    return np.array([(int(f.split('_')[0].split('t00000')[1]), 
                      load_graph(d+f)) for f in files])

def get_inputs(graphs):
    size = len(graphs)
    O  = [Variable(torch.FloatTensor(graphs[i].X))
          for i in range(size)]
    Rs = [Variable(torch.FloatTensor(graphs[i].Ro))
          for i in range(size)]
    Rr = [Variable(torch.FloatTensor(graphs[i].Ri))
          for i in range(size)]
    Ra = [Variable(torch.FloatTensor(graphs[i].a)).unsqueeze(0)
          for i in range(size)]
    y  = [Variable(torch.FloatTensor(graphs[i].y)).unsqueeze(0).t()
          for i in range(size)]
    return O, Rs, Rr, Ra, y

def plotLosses(test_losses, train_losses, name):
    plt.plot(test_losses,  color='mediumslateblue', label="Training",
             marker='h', lw=0, ms=1.4)
    plt.plot(train_losses, color='mediumseagreen',  label="Testing",
             marker='h', lw=0, ms=1.4)
    plt.xlabel("Epoch")
    plt.ylabel("RMS Loss")
    plt.legend(loc='upper right')
    plt.savefig(name, dpi=1200)
    plt.clf()

# grab config parameters
args = parse_args()
with open(args.config) as f:
    config = yaml.load(f, yaml.FullLoader)

model_outdir = config['model_outdir']
plot_outdir  = config['plot_outdir']
verbose = config['verbose']
prep, pt_cut = config['prep'], config['pt_cut']
n_epoch, n_batch = config['n_epoch'], config['n_batch']
save_every = config['save_every_n_epoch']
save_last  = config['save_last_n_epoch']

# job name, ex. "LP_0p5_1200"
job_name = "{0}_{1}_{2}".format(prep, pt_cut, n_epoch)

if (verbose):
    print('\nBeginning', job_name)
    print(' --> Writing models to', model_outdir)
    print(' --> Writing plots to', plot_outdir)

# pull 1000 graphs from the train_1 sample
graph_dir = "/tigress/jdezoort/IN_samples_large/IN_{0}_{1}/".format(prep, pt_cut)
graphs = get_graphs(graph_dir)

# objects: (r, phi, z); relations: (0); effects: (weight) 
object_dim, relation_dim, effect_dim = 3, 1, 1
interaction_network = InteractionNetwork(object_dim, relation_dim, effect_dim)
optimizer = optim.Adam(interaction_network.parameters())
criterion = nn.MSELoss()

# config mini-batch
train_size, test_size = 800, 200
batch_size = int(float(train_size)/n_batch)
if (verbose): print(" --> Mini-Batch: batch_size={0}".format(batch_size))

# prepare test graphs
test_graphs = [graphs[(train_size-1)+i][1] for i in range(test_size)]
test_O, test_Rs, test_Rr, test_Ra, test_y = get_inputs(test_graphs)

save_epochs = np.arange(0, n_epoch, save_every)
if (verbose): print(" --> Saving the following epochs:\n", save_epochs)

test_losses, train_losses, batch_losses = [], [], []
for epoch in range(n_epoch):
    print("Epoch #", epoch)
    batch_loss = 100
    
    # loss computed on a per-batch basis
    for b in range(n_batch):
        rand_idx  = [random.randint(0, train_size) for _ in range(batch_size)]
        batch_of_graphs = [graphs[i][1] for i in rand_idx]

        O, Rs, Rr, Ra, y = get_inputs(batch_of_graphs)

        predicted = interaction_network(O, Rs, Rr, Ra)
        loss = criterion(torch.cat(predicted, dim=0), torch.cat(y, dim=0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss = np.sqrt(loss.data)
        batch_losses.append(batch_loss)
    
    predicted = interaction_network(test_O, test_Rs, test_Rr, test_Ra)
    loss = criterion(torch.cat(predicted, dim=0), torch.cat(test_y, dim=0))

    train_losses.append(batch_loss)
    test_losses.append(np.sqrt(loss.data))

    if epoch in save_epochs:
        outfile = "{0}/{1}_epoch{2}.pt".format(model_outdir, job_name, epoch)
        torch.save(interaction_network.state_dict(), outfile)

plotLosses(test_losses, train_losses, "{0}/losses_{1}.png".format(plot_outdir, job_name))

