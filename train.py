import os
import sys
import platform
import datetime
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable

import matplotlib
if os.environ.get('DISPLAY','') == '':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.axes as axs
import numpy as np

from PPIPointNet import PointNet, DualPointNet
from evaluate import evaluateModel
from dataset import PDBset, DualPDBset
from utils import get_lr, saveModel, FavorHighLoss

# PRINT INFORMATION

time = datetime.datetime.utcnow()

print('ABOUT')
print('    Simplified PointNet for Protein-Protein Reaction - Training script')
print('    Lukas De Clercq, 2018, Netherlands eScience Center\n')
print('    See attached license')

print('RUNTIME INFORMATION')
print('    System    -', platform.system(), platform.release(), platform.machine())
print('    Version   -', platform.version())
print('    Node      -', platform.node())
print('    Time      -', time, 'UTC', '\n')

print('LIBRARY VERSIONS')
print('    Python    -', platform.python_version(), 'on', platform.python_compiler())
print('    Pytorch   -', torch.__version__)
print('    CUDA      -', torch.version.cuda)
print('    CUDNN     -', torch.backends.cudnn.version(), '\n', flush = True)

# ---- OPTION PARSING ----

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256, help='Input batch size (default = 256)')
parser.add_argument('--num_points', type=int, default=1024, help='Points per point cloud used (default = 1024)')
parser.add_argument('--num_epoch',  type=int,  default=15, help='Number of epochs to train for (default = 15)')
parser.add_argument('--CUDA',       dest='CUDA', default=False, action='store_true', help='Train on GPU')
parser.add_argument('--out_folder', type=str, default='~',  help='Model output folder')
parser.add_argument('--model',      type=str, default='',   help='Model input path')
parser.add_argument('--data_path',  type=str, default='~', help='Path to HDF5 file')
parser.add_argument('--lr',         type=float, default=0.0001, help='Learning rate (default = 0.0001)')
parser.add_argument('--optimizer',  type=str, default='Adam', help='What optimizer to use. Options: Adam, SGD, SGD_cos')
parser.add_argument('--avg_pool',   dest='avg_pool', default=False, action='store_true', help='Use average pooling after for feature pooling (instead of default max pooling)')
parser.add_argument('--dual',       dest='dual', default=False, action='store_true', help='Use DualPointNet architecture')
parser.add_argument('--get_min',    dest='get_min', default=False, action='store_true', help='Get minimum point cloud size')
parser.add_argument('--metric',     type=str, default='dockQ',   help='Metric to be used. Options: irmsd, lrmsd, fnat, dockQ (default)')
parser.add_argument('--dropout',    type=float, default=0.5, help='Dropout rate in last layer. When 0 replaced by batchnorm (default = 0.3)')


arg = parser.parse_args()
print('RUN PARAMETERS')
print('    ', arg, '\n', flush=True)

# ---- DATA LOADING ----

if arg.dual:
    dataset = DualPDBset(hdf5_file=arg.data_path, group='train', num_points=arg.num_points, metric=arg.metric)
    testset = DualPDBset(hdf5_file=arg.data_path, group='test', num_points=arg.num_points, metric=arg.metric)
else:
    dataset = PDBset(hdf5_file=arg.data_path, group='train', num_points=arg.num_points, metric=arg.metric)
    testset = PDBset(hdf5_file=arg.data_path, group='test', num_points=arg.num_points, metric=arg.metric)

dataloader = data.DataLoader(dataset, batch_size=arg.batch_size, shuffle=True, num_workers=1)
testloader = data.DataLoader(testset, batch_size=arg.batch_size, shuffle=True, num_workers=1)

num_batch = len(dataset)/arg.batch_size

print('DATA PARAMETERS')
print('    Test & train sizes: %d & %d -> %.1f' %(len(testset), len(dataset), 100*len(testset)/len(dataset)), '%', flush=True)

# ---- GET MINIMUM

if arg.get_min:
    minSize = min(dataset.getMin(), testset.getMin())
    print('    Minimum pointcloud size:', minSize)
print('')

# ---- SET UP MODEL ----

# Architecture selection

if arg.metric == 'dockQ':
    sigmoid = True
else:
    sigmoid = False

if arg.dual:
    net = DualPointNet(num_points=arg.num_points, in_channels=dataset.getFeatWidth(), avgPool=arg.avg_pool, sigmoid=sigmoid, dropout=arg.dropout)
else:
    net = PointNet(num_points=arg.num_points, in_channels=dataset.getFeatWidth(), avgPool=arg.avg_pool, sigmoid=sigmoid, dropout=arg.dropout)

# Model loading (continued/transfer learning)
if arg.model != '':
    net.load_state_dict(torch.load(arg.model)) 

# GPU  & GPu parallellization
if arg.CUDA:
    net.cuda()
    model = torch.nn.DataParallel(net) 
else:
    model = net

# Optimizer selection
if arg.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=arg.lr)
elif arg.optimizer == 'SGD' or arg.optimizer == 'SGD_cos':
    optimizer = optim.SGD(model.parameters(), lr=arg.lr, momentum=0.9)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_batch)

# Loss function
train_loss_func = FavorHighLoss()
test_loss_func = FavorHighLoss(size_average=False)

# ---- MODEL TRAINING ----

print('START TRAINING')
model.train()  # Set to training mode

prev_test_score,x1,y1 = evaluateModel(model, test_loss_func, testloader, arg.dual, arg.CUDA)
print('    Before training - Test loss = %.5f\n' %(prev_test_score))
print('    WARNING: Train loss is with the model in eval mode, this')
print('             alters dropout and batchnorm behaviour. Train loss')
print('             can be expected to be worse under these conditions')

early_stop_count = 0

# Main epoch loop
for epoch in range(arg.num_epoch):
    avg_train_score = 0

    # Loss rate scheduling
    if arg.optimizer == 'SGD_cos':
        scheduler.base_lrs = [arg.lr*(1-(epoch**2)/(arg.num_epoch**2))]
        scheduler.step(epoch=0)

    # Iterate over minibatches
    for i, data in enumerate(dataloader, 0):
        optimizer.zero_grad()

        # Data loading & manipulation
        points, target = data
        points, target = Variable(points), Variable(target)  # Deprecated in PyTorch >=0.4
        points = points.transpose(2, 1)
        if arg.CUDA:
            points, target = points.cuda(), target.cuda()

        # No partial last batches, in order to reduce noise in gradient.
        if len(target) != arg.batch_size:
            break 

        # Forward and backward pass
        prediction = model(points).view(-1)
        loss = train_loss_func(prediction, target)
        avg_train_score += loss
        loss.backward()
        print('    E: %02d - %02d/%02d - LR: %.6f - Loss: %.5f' %(epoch+1, i+1, num_batch, get_lr(optimizer)[0], loss), flush=True,  end="\r")

        # Stepping
        optimizer.step()
        if arg.optimizer == 'SGD_cos':
            scheduler.step()

    # This section runs at the end of each batch
    test_score,x1,y1 = evaluateModel(model, test_loss_func, testloader, arg.dual, arg.CUDA)
    print('    E: %02d - Train loss = %.5f              ' %(epoch+1, avg_train_score/num_batch))
    print('    E: %02d - Test loss = %.5f\n' %(epoch+1, test_score))

    # Early stopping
    if test_score > prev_test_score:
        early_stop_count += 1
        if early_stop_count == 3:
            print('    Early stopping condition reached')
            break 
    else:
        early_stop_count = 0
        saveModel(model, arg, str(time))
        prev_test_score = test_score

# ---- REVERT TO BEST MODEL ----

print('    Reverting to best known model (test loss = %.5f)\n' %prev_test_score)    
model.load_state_dict(torch.load('%s/PPIPointNet.pth' % (arg.out_folder))) # Load best known configuration

# ---- PLOTTING ----

print('CREATING PLOT')
train_score,x2,y2 = evaluateModel(model, test_loss_func, dataloader, arg.dual, arg.CUDA)
print('    Final train loss = %.5f' %(train_score))

print('    Creating plot...')
fig, ax = plt.subplots()
ax.scatter(x2.data.cpu(),y2.data.cpu(), label='Train',s=1)
ax.scatter(x1.data.cpu(),y1.data.cpu(), label='Test',s=1)
ax.set_ylabel('Prediction')
ax.set_xlabel('Truth')
ax.set_xlim(xmin=0.0)
ax.set_ylim(ymin=0.0)
ax.legend(loc='best')
title = 'Test loss: %.5f' %prev_test_score
fig.suptitle(title)
fig.set_size_inches(19.2, 10.8) # 1920 x 1080 when using 100 dpi
fig.savefig('post-train.png', dpi=100)