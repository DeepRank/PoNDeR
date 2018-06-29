# General

Experimental deep learning architecture for scoring protein-protein interactions.

See [PointNet paper](https://arxiv.org/abs/1612.00593) for original architecture description. This implementation differs does not contain the transformer networks, so can be considered the *vanilla* version of PointNet.

Other adaptations include cosine annealing learning rate decay, which has been implemented to improve accuracy and generalizability of the trained network. See [Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983).

All layers receive batch normalization, except the last linear layer where 30% dropout is applied, again to make the final network generalize better.

# Dependencies

* Python 3.x
* Model & training: [PyTorch <0.4](https://github.com/pytorch/pytorch) and its dependencies
* Data conversion: [DeepRank](https://github.com/DeepRank/deeprank) and its dependencies

# Current state

* Architecture & training scripts have been fully implemented
* Initial test show promise and generalizability
* Tests on larger dataset are WIP