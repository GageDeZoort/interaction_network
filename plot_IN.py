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
import plots.plot_menu as pm

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('train_IN.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/train_LP.yaml')
    return parser.parse_args()
    
# grab config parameters
args = parse_args()
with open(args.config) as f:
    config = yaml.load(f, yaml.FullLoader)

prep      = config['prep']
pt_cut    = config['pt_cut']
n_epoch   = config['n_epoch']
n_batch   = config['n_batch']
model_dir = config['model_outdir']
train_size, test_size = 800, 200

# job name, ex. "LP_0p5_1600_32"
job_name = "{0}_{1}_{2}".format(prep, pt_cut, n_epoch)

# get model paths
models = os.listdir(model_dir)
models = ["{0}/{1}".format(model_dir, model) 
          for model in models if job_name in model]

# load in test graph paths
graph_dir = "/tigress/jdezoort/IN_samples_large/IN_{0}_{1}/".format(prep, pt_cut)
graph_files = os.listdir(graph_dir)
test_graphs = np.array([(int(f.split('_')[0].split('t00000')[1]),
                         load_graph(graph_dir+f)) for f in graph_files[train_size:]])

# prepare test graphs
size = len(test_graphs)
test_O  = [Variable(torch.FloatTensor(test_graphs[i][1].X))
           for i in range(size)]
test_Rs = [Variable(torch.FloatTensor(test_graphs[i][1].Ro))
           for i in range(size)]
test_Rr = [Variable(torch.FloatTensor(test_graphs[i][1].Ri))
           for i in range(size)]
test_Ra = [Variable(torch.FloatTensor(test_graphs[i][1].a)).unsqueeze(0)
           for i in range(size)]
test_y  = [Variable(torch.FloatTensor(test_graphs[i][1].y)).unsqueeze(0).t()
           for i in range(size)]

# build IN
interaction_network = InteractionNetwork(3, 1, 1)
interaction_network.load_state_dict(torch.load(models[-1]))
interaction_network.eval()
predicted = interaction_network(test_O, test_Rs, test_Rr, test_Ra)

# shape up data for analysis
predicted = torch.cat(predicted, dim=0)
predicted = np.array([float(predicted[i].data[0])
                      for i in range(len(predicted))])
test_y = torch.cat(test_y, dim=0)
test_y = np.array([float(test_y[i].data[0])
                   for i in range(len(test_y))])

real_seg_idx = (test_y==1).nonzero()[:][0]
real_seg = predicted[real_seg_idx]
fake_seg_idx = (test_y==0).nonzero()[:][0]
fake_seg = predicted[fake_seg_idx]

# order some plots
pm.plotDiscriminant(real_seg, fake_seg, 20, "plots/discriminant_{0}.png".format(job_name))

for i in np.arange(0, 1, 0.01):
    testConfusion = pm.confusionMatrix(real_seg, fake_seg, i)
    print(i, testConfusion)

pm.confusionPlot(real_seg, fake_seg, "plots/confusions_{0}.png".format(job_name))
pm.plotROC(real_seg, fake_seg, "plots/ROC_{0}.png".format(job_name))
