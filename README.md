# interaction_network
## Overview
This repo is based on interaction network (IN) model [1], a graph neural network (GNN) architecture that applies relational and object models in stages to infer abstract interactions and object dynamics. 

## Dataset 
This model is tested and trained with events from the [Kaggle TrackML dataset](https://www.kaggle.com/c/trackml-particle-identification). TrackML simulates high pileup tracks inside a general-purpose particle tracker, supplying simulated hits and particles, as well as ground truth information. 

## Pre-processing
Each TrackML event is converted into a directed multigraph of hits connected by segments. Different pre-processing strategies are available, each with different graph construction efficiencies. This repo contains two such stratigies:
   1) Select one hit per particle per layer, connect hits in adjacent layers. This is the strategy used by the HEP.TrkX collaboration, which we denote "layer pairs" (see **prep_LP.py**). 
   2) Select hits between adjacent layers and hits within the same layer, requiring that same-layer hits are within some distance dR of each other. 

## Models

The code in the repo is organized as follows:
* **model/graph.py**: defines a graph as a namedTuple containing:
    * **x**:   a size (N<sub>hits</sub> x 3) feature vector for each hit, containing the hit coordinates (r, phi, z)
    * **R_i**: a size (N<sub>hits</sub> x N<sub>segs</sub>) matrix whose entries (**R_i**)<sub>hs</sub> are 1 when segment s is incoming to hit h, and 0 otherwise
    * **R_o**: a size (N<sub>hits</sub> x N<sub>segs</sub>) matrix whose entries (**R_i**)<sub>hs</sub> are 1 when segment s is outgoing from hit h, and 0 otherwise
    * **a**:   a size (N<sub>segs</sub> x 1) vector whose s<sup>th</sup> entry is 0 if the segment s connects opposite-layer hits and 1 if segment s connects same-layer hits
*   **model/interaction_network.py**: produces edge weights for each segment by applying a relational model to the hit/segment interactions, aggregating the resulting effects for each receiving hit, re-embedding the hit features with an object model, and re-applying the relational model to each interaction
*   **model/relational_model.py**:
## References
[1] “Interaction networks for learning about objects relations and physics”, [arXiv:1612.00222](https://arxiv.org/abs/1612.00222)
