# General

See [PointNet paper](https://arxiv.org/abs/1612.00593) for architecture description. This implementation does not contain the transformer networks, so can be considered the *vanilla* version of PointNet.

Cosine annealing, as well as step learning rate decay, has been implemented to improve generalizability of the trained network.

All layers receive batch normalization, except the last linear layer where 30% dropout is applied, again to make the final network generalize better.

# Current state

Architecture & training scripts have been fully defined and tested.
Protein datasets are yet to be implemented, see TODO's in file. 