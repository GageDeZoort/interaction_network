import os
import random
import time
import argparse

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
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('-n', '--nEpochs', default=1, type=int)
    add_arg('-p', '--prep', default='LP', type=str)
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

def plotDiscriminant(true_seg, false_seg, nBins, filename):
    plt.hist(true_seg, nBins, color='blue', label='Real Segments', alpha=0.7)
    plt.hist(false_seg, nBins, color='red', label='Fake Segments', alpha=0.7)
    plt.xlim(0, 1)
    plt.xlabel("Edge Weight")
    plt.ylabel("Counts")
    plt.title("Edge Weights: All Segments")
    plt.legend(loc="upper right")
    plt.savefig(filename, dpi=1200)
    plt.clf()

def confusionMatrix(true_seg, false_seg, cut):
    true_seg, false_seg = np.array(true_seg), np.array(false_seg)
    n_t  = len(true_seg)
    n_f  = len(false_seg)
    TP = len(true_seg[true_seg > cut])
    FN = len(true_seg[true_seg < cut])
    FP = len(false_seg[false_seg > cut])
    TN = len(false_seg[false_seg < cut])
    return np.array([[TP, FN], [FP, TN]])

def printConfusionReport(matrix, cutoff):
    print("Confusion Matrix", cutoff)
    print("TP =", testConfusion[0][0])
    print("FN =", testConfusion[0][1])
    print("FP =", testConfusion[1][0])
    print("TN =", testConfusion[1][1])
    P = float(testConfusion[0][0] + testConfusion[0][1])
    N = float(testConfusion[1][0] + testConfusion[1][1])
    print("TPR =", testConfusion[0][0]/P)
    print("FNR =", testConfusion[0][1]/P)
    print("FPR =", testConfusion[1][0]/N)
    print("TNR =", testConfusion[1][1]/N)    
    
def confusionPlot(true_seg, false_seg, name):
    cuts = np.arange(0, 1, 0.01)
    matrices = [confusionMatrix(true_seg, false_seg, i) for i in cuts]
    P = float(matrices[0][0][0] + matrices[0][0][1])
    N = float(matrices[0][1][0] + matrices[0][1][1])
    TPR = [matrices[i][0][0]/P for i in range(len(cuts))]
    TNR = [matrices[i][1][1]/N for i in range(len(cuts))]
    FNR = [matrices[i][0][1]/P for i in range(len(cuts))]
    FPR = [matrices[i][1][0]/N for i in range(len(cuts))]
    
    plt.scatter(cuts, TPR, label="True Positive", 
                color='mediumseagreen', marker='h', s=1.8)
    plt.scatter(cuts, FNR, label="False Negative",
                color='orange', marker='h', s=1.8)
    plt.scatter(cuts, FPR, label="False Positive",
                color='red', marker='h', s=1.8)
    plt.scatter(cuts, TNR, label="True Negative",
                color='mediumslateblue', marker='h', s=1.8)
    plt.xlabel('Discriminant Cut')
    plt.ylabel('Yield')
    plt.legend(loc='best')
    plt.savefig(name, dpi=1200)
    plt.clf()

    plt.scatter(FPR, TPR, color='mediumslateblue')
    plt.grid(True)
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig("plots/ROC.png")
    plt.clf()

args = parse_args()
prep = args.prep

object_dim = 3     # (r, phi, z)
relation_dim = 1   
effect_dim = 1     # edge weights

graph_dir = "/tigress/jdezoort/IN_samples_large/IN_" + prep + "_2/"
graphs = get_graphs(graph_dir)

interaction_network = InteractionNetwork(object_dim, relation_dim, effect_dim)
optimizer = optim.Adam(interaction_network.parameters())
criterion = nn.MSELoss()

n_epoch = args.nEpochs
batches_per_epoch = 25
batch_size = 32

test_size = 200
test_graphs = [graphs[799+i][1] for i in range(test_size)]
test_O, test_Rs, test_Rr, test_Ra, test_y = get_inputs(test_graphs)

test_losses = []
train_losses = []
batch_losses = []

for epoch in range(n_epoch):
    print("Epoch #", epoch)
    batch_loss = 100

    for b in range(batches_per_epoch):
        rand_idx  = [random.randint(0, 800) for _ in range(batch_size)]
        batch_of_graphs = [graphs[i][1] for i in rand_idx]

        O, Rs, Rr, Ra, y = get_inputs(batch_of_graphs)

        predicted = interaction_network(O, Rs, Rr, Ra)
        loss = criterion(torch.cat(predicted, dim=0), torch.cat(y, dim=0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss = np.sqrt(loss.data)
        batch_losses.append(batch_loss)
        
    train_losses.append(batch_loss)
    predicted = interaction_network(test_O, test_Rs, test_Rr, test_Ra)
    loss = criterion(torch.cat(predicted, dim=0),
                     torch.cat(test_y, dim=0))
    test_losses.append(np.sqrt(loss.data))


plotLosses(test_losses, train_losses, "plots/losses_" + prep + ".png")

predicted = interaction_network(test_O, test_Rs, test_Rr, test_Ra)

true_seg,    false_seg    = [], []
true_seg_SL, false_seg_SL = [], [] # same-layer
true_seg_DL, false_seg_DL = [], [] # different-layer

for i in range(test_size):
    result = predicted[i]
    target = test_y[i]
    sameLayer = test_Ra[i].t()
    
    for j in range(len(test_y[i])):
        if (target[j] == 0):
            false_seg.append(float(result[j]))
            if (sameLayer[j] == 1):
                false_seg_SL.append(float(result[j]))
            elif (sameLayer[j] == 0):
                false_seg_DL.append(float(result[j]))
        if (target[j] == 1):
            true_seg.append(float(result[j]))
            if (sameLayer[j] == 1):
                true_seg_SL.append(float(result[j]))
            elif (sameLayer[j] == 0):
                true_seg_DL.append(float(result[j]))
        
plotDiscriminant(true_seg, false_seg, 20, "plots/discriminant_"+prep+".png")

if (prep == "LPP"):
    plotDiscriminant(true_seg_SL, false_seg_SL, 20, "plots/discriminant_SL_"+prep+".png")
    plotDiscriminant(true_seg_DL, false_seg_DL, 20, "plots/discriminant_DL_"+prep+".png")

for i in np.arange(0, 1, 0.01):
    testConfusion = confusionMatrix(true_seg, false_seg, i)
    printConfusionReport(testConfusion, i)

confusionPlot(true_seg, false_seg, "plots/confusions_"+prep+".png")
