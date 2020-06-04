# interaction_network
## Overview
This repo is based on interaction network (IN) model [1], a graph neural network (GNN) architecture that applies relational and object models in stages to infer abstract interactions and object dynamics. 

## Dataset 
This model is tested and trained with events from the [Kaggle TrackML dataset](https://www.kaggle.com/c/trackml-particle-identification). TrackML simulates high pileup tracks inside a general-purpose particle tracker, supplying simulated hits and particles, as well as ground truth information. 

## Pre-processing
Each TrackML event is converted into a directed multigraph of hits connected by segments. Different pre-processing strategies are available, each with different graph construction efficiencies. This repo contains two such stratigies:
   1) Select one hit per particle per layer, connect hits in adjacent layers. This is the strategy used by the [HEP.TrkX collaboration](https://heptrkx.github.io/), which we denote "layer pairs" (see **prep_LP.py**) [2]. 
   2) Select hits between adjacent layers *and* hits within the same layer, requiring that same-layer hits are within some distance dR of each other (see **prep_LPP.py**).

## Models
The code in the repo is organized as follows:
* **model/graph.py**: defines a graph as a namedTuple containing:
    * **x**:   a size (N<sub>hits</sub> x 3) feature vector for each hit, containing the hit coordinates (r, phi, z)
    * **R_i**: a size (N<sub>hits</sub> x N<sub>segs</sub>) matrix whose entries (**R_i**)<sub>hs</sub> are 1 when segment s is incoming to hit h, and 0 otherwise
    * **R_o**: a size (N<sub>hits</sub> x N<sub>segs</sub>) matrix whose entries (**R_i**)<sub>hs</sub> are 1 when segment s is outgoing from hit h, and 0 otherwise
    * **a**:   a size (N<sub>segs</sub> x 1) vector whose s<sup>th</sup> entry is 0 if the segment s connects opposite-layer hits and 1 if segment s connects same-layer hits
*   **model/interaction_network.py**: produces edge weights for each segment by applying a relational model to the hit/segment interactions, aggregating the resulting effects for each receiving hit, re-embedding the hit features with an object model, and re-applying the relational model to each interaction
*   **model/relational_model.py**: a MLP that outputs 1 parameter per segment, which we interpret as an edge weight (truth probability)
*   **model/object_model.py**: a MLP that outputs 3 parameters per hit, which are the re-embedded position featatures of the hit

## Training
The script **train_IN.py** builds and trains an IN on pre-processed TrackML graphs. The training data is organzied into mini-batches, which are used to optimize an RMS loss function via the adam optimizer. The IN is tested on a separate sample of pre-processed graphs, producing output including confusion matrices, a ROC curve, TPR/TNR/FNR/FPR curves, and a discriminant plot. 

## Slurm Scripts
The **prep_IN.slurm** and **train_IN.slurm** scripts submit the pre-processing and training scripts as jobs to a slurm-based job submission system.

## To-Do
* Move the plotting stage of **train_IN.py** to an external script
* Optimize the discriminant calculations
* Write the mini-batches in native PyTorch

## References
[1] “Interaction networks for learning about objects relations and physics”, [arXiv:1612.00222](https://arxiv.org/abs/1612.00222)
[2] HEP.TrkX [Layer Pairs Implementation](https://github.com/HEPTrkX/heptrkx-gnn-tracking/blob/master/prepare.py)
