# General

Experimental deep learning architecture for scoring protein-protein interactions.

See [PointNet paper](https://arxiv.org/abs/1612.00593) for original architecture description. This implementation contains two architectures, neither of which contain the transformer networks, so can be considered variants of the *vanilla* version of PointNet. The first differs merely in its dropout rate (50%), whereas the second is a novel architecture called *Siamese PointNet*, visible in the image below.

![Siamese PointNet](/doc/SiamesePointNet_architecture.png)

Other adaptations include cosine annealing learning rate decay, which has been implemented to improve accuracy and generalizability of the trained network (see [Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)), and a custom loss function introducing a bias in learning towards higher scoring decoys.

![FavorHigh loss](/doc/FavorHighLoss.png)

# Dependencies

* Python 3.x
* [H5Py](http://www.h5py.org/) for fast data retrieval
* [PyTorch <0.4](https://github.com/pytorch/pytorch) and its dependencies
* Data conversion uses [DeepRank](https://github.com/DeepRank/deeprank) and its dependencies
* [Seaborn](https://github.com/mwaskom/seaborn) for plotting

# Usage

*python train.py*

```
  --batch_size BATCH_SIZE   Input batch size (default = 256)
  --num_points NUM_POINTS   Points per point cloud used (default = 1024)
  --num_epoch NUM_EPOCH     Number of epochs to train for (default = 15)
  --CUDA                    Train on GPU
  --out_folder OUT_FOLDER   Model output folder
  --model MODEL             Model input path
  --data_path DATA_PATH     Path to HDF5 file
  --lr LR                   Learning rate (default = 0.0001)
  --optimizer OPTIMIZER     What optimizer to use. Options: Adam, SGD, SGD_cos
  --avg_pool                Use average pooling for feature pooling (instead of default max pooling)
  --dual                    Use Siamese PointNet architecture
  --metric METRIC           Metric to be used. Options: irmsd, lrmsd, fnat, dockQ (default)
  --dropout DROPOUT         Dropout rate in last layer. When 0 replaced by batchnorm (default = 0.5)
  --root                    Apply square root on metric (for DockQ score balancing)
  --patience PATIENCE       Number of epochs to observe overfitting before early stopping
  --classification          Classification instead of regression
```

The network takes the atoms taking part in an interaction as point cloud data. Data conversion can be performed using the *extract_pc.py* script.

Data is saved in HDF5 format containing 3 groups: train, test and "holdout" data. Datasets within these groups contain atom features with *float32* precision and attributes containing the iRMSD, lRMSD, FNAT, and DockQ scores.

# Current state

* Architecture & training scripts have been fully implemented
* Multiple tests have shown inadequacy of all tested architectures. It is deemed unlikely that PointNet or PointNet derivatives can produce an acceptable scoring model.