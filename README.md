# interaction_network
## Implementation
The code is stored in the Tiger cluster, and can be found here: 

**/home/jdezoort/GNN_Repo_Reorganization/gnn_track_challenge/interaction_network**

The code in the repo is organized as follows:
* **Model/Graph.py**: defines a graph as a namedTuple containing:
    * **x**:   a size (N<sub>hits</sub> x 3) feature vector for each hit, containing the hit coordinates (r, phi, z)
    * **R_i**: a size (N<sub>hits</sub> x N<sub>segs</sub>) matrix whose entries (**R_i**)<sub>hs</sub> are 1 when segment s is incoming to hit h, and 0 otherwise
    * **R_o**: a size (N<sub>hits</sub> x N<sub>segs</sub>) matrix whose entries (**R_i**)<sub>hs</sub> are 1 when segment s is outgoing from hit h, and 0 otherwise
    * **a**:   a size (N<sub>segs</sub> x 1) vector whose s<sup>th</sup> entry is 0 if the segment s connects opposite-layer hits and 1 if segment s connects same-layer hits
*   **Model/InteractionNetwork.py**: 
