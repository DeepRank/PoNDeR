import os
import sys
import platform
import datetime
import argparse
from timeit import default_timer as timer
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable

from PPIPointNet import PointNet, DualPointNet
from evaluate import evaluateModel
from dataset import PDBset, DualPDBset
from utils import get_lr, saveModel, FavorHighLoss, calcAccuracy
from plotLoss import scatter

time = datetime.datetime.now()

# ---- OPTION PARSING ----

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256, help='Input batch size (default = 256)')
parser.add_argument('--num_points', type=int, default=1024, help='Points per point cloud used (default = 1024)')
parser.add_argument('--num_epoch',  type=int,  default=15, help='Number of epochs to train for (default = 15)')
parser.add_argument('--CUDA',       dest='CUDA', default=False, action='store_true', help='Train on GPU')
parser.add_argument('--out_folder', type=str, default=str(Path.home()),  help='Model output folder')
parser.add_argument('--model',      type=str, default='',   help='Model input path')
parser.add_argument('--data_path',  type=str, default='~', help='Path to HDF5 file')
parser.add_argument('--lr',         type=float, default=0.0001, help='Learning rate (default = 0.0001)')
parser.add_argument('--optimizer',  type=str, default='Adam', help='What optimizer to use. Options: Adam, SGD, SGD_cos')
parser.add_argument('--avg_pool',   dest='avg_pool', default=False, action='store_true', help='Use average pooling after for feature pooling (instead of default max pooling)')
parser.add_argument('--dual',       dest='dual', default=False, action='store_true', help='Use DualPointNet architecture')
parser.add_argument('--metric',     type=str, default='dockQ',   help='Metric to be used. Options: irmsd, lrmsd, fnat, dockQ (default)')
parser.add_argument('--dropout',    type=float, default=0.5, help='Dropout rate in last layer. When 0 replaced by batchnorm (default = 0.5)')
parser.add_argument('--log',        dest='log', default=False, action='store_true', help='Apply logarithm on metric')
parser.add_argument('--patience',   type=int, default=5, help='Number of epochs to observe overfitting before early stopping')
parser.add_argument('--classification',dest='classification', default=False, action='store_true', help='Classification instead of regression')

arg = parser.parse_args()

save_path = arg.out_folder+'/'+time.strftime('%d%m-%H%M')

if not os.path.exists(save_path):
    os.makedirs(save_path)

# ---- DATA LOADING ----

if arg.dual:
    dataset = DualPDBset(hdf5_file=arg.data_path, group='train', num_points=arg.num_points, metric=arg.metric, log=arg.log, classification=arg.classification)
    testset = DualPDBset(hdf5_file=arg.data_path, group='test', num_points=arg.num_points, metric=arg.metric, log=arg.log, classification=arg.classification)
else:
    dataset = PDBset(hdf5_file=arg.data_path, group='train', num_points=arg.num_points, metric=arg.metric, log=arg.log, classification=arg.classification)
    testset = PDBset(hdf5_file=arg.data_path, group='test', num_points=arg.num_points, metric=arg.metric, log=arg.log, classification=arg.classification)

dataloader = data.DataLoader(dataset, batch_size=arg.batch_size, shuffle=True, num_workers=1)
testloader = data.DataLoader(testset, batch_size=arg.batch_size, shuffle=True, num_workers=1)

num_batch = len(dataset)/arg.batch_size

# ---- PRINT INFORMATION ----
print('ABOUT')
print('    Simplified PointNet for Protein-Protein Reaction')
print('    Lukas De Clercq, 2018, Netherlands eScience Center')
print('    See attached license\n')

print('RUNTIME INFORMATION')
print('    System    -', platform.system(), platform.release(), platform.machine())
print('    Version   -', platform.version())
print('    Node      -', platform.node())
print('    Time      -', time, 'UTC', '\n')

print('LIBRARY VERSIONS')
print('    Python    -', platform.python_version(), 'on', platform.python_compiler())
print('    Pytorch   -', torch.__version__)
print('    CUDA      -', torch.version.cuda)
print('    CUDNN     -', torch.backends.cudnn.version(), '\n')

print('RUN PARAMETERS')
for a in vars(arg):
    print('    ', a, getattr(arg, a))
print('')

print('DATA PARAMETERS')
print('    Test & train sizes: %d & %d -> %.1f' %(len(testset), len(dataset), 100*len(testset)/len(dataset)), '%')

if arg.classification:
    targets = []
    for data in testloader:
            _, target = data
            targets.append(np.array(target))
    targets = np.concatenate(targets)
    pos = 100*sum(targets)/len(targets)
    print('    Positive samples: %.1f' %(pos), '%')

# ---- SET UP MODEL ----

print('\nMODEL SET-UP')

# Architecture selection

if arg.metric == 'dockQ' and not arg.classification:
    sigmoid = True
else:
    sigmoid = False

if arg.dual:
    net = DualPointNet(num_points=arg.num_points, in_channels=dataset.getFeatWidth(), avgPool=arg.avg_pool, sigmoid=sigmoid, dropout=arg.dropout, classification=arg.classification)
else:
    net = PointNet(num_points=arg.num_points, in_channels=dataset.getFeatWidth(), avgPool=arg.avg_pool, sigmoid=sigmoid, dropout=arg.dropout, classification=arg.classification)

# GPU  & GPU parallellization
if arg.CUDA:
    net.cuda()
    model = torch.nn.DataParallel(net) 
else:
    model = net

print(model)

# Model loading (continued/transfer learning)
if arg.model != '':
    model.load_state_dict(torch.load(arg.model)) 

# Optimizer selection
if arg.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=arg.lr)
elif arg.optimizer == 'SGD' or arg.optimizer == 'SGD_cos':
    optimizer = optim.SGD(model.parameters(), lr=arg.lr, momentum=0.9)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_batch)

# Loss function
if arg.classification:
    train_loss_func = nn.CrossEntropyLoss()
    test_loss_func = nn.CrossEntropyLoss(size_average=False)
elif arg.log:
    train_loss_func = nn.L1Loss()
    test_loss_func = nn.L1Loss(size_average=False)
else:
    train_loss_func = FavorHighLoss()
    test_loss_func = FavorHighLoss(size_average=False)

# ---- MODEL TRAINING ----
print('\nTRAINING')
model.train()  # Set to training mode

prev_test_score,x1,y1 = evaluateModel(model, test_loss_func, testloader, arg.dual, arg.CUDA, classification=arg.classification)
print('    Before training - Test loss = %.5f' %(prev_test_score))
if arg.classification:
    acc = calcAccuracy(x1,y1)
    print('                      Test accuracy = %.2f' %(acc), '%')
print('\n    WARNING: Train loss is with the model in eval mode, this alters dropout and batchnorm')
print('             behaviour. Train loss can be expected to be worse under these conditions\n')

early_stop_count = 0
avg_time_per_epoch = 0


# Main epoch loop
for epoch in range(arg.num_epoch):
    start = timer()
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
        prediction = model(points)
        if not arg.classification:
            prediction = prediction.view(-1)
        loss = train_loss_func(prediction, target)
        avg_train_score += loss
        loss.backward()
        print('E: %02d - %02d/%02d - LR: %.6f - Loss: %.5f' %(epoch+1, i+1, num_batch, get_lr(optimizer)[0], loss), flush=True,  end='\r')

        # Stepping
        optimizer.step()
        if arg.optimizer == 'SGD_cos':
            scheduler.step()

    # This section runs at the end of each batch
    test_score,x1,y1 = evaluateModel(model, test_loss_func, testloader, arg.dual, arg.CUDA, classification=arg.classification)
    print('E: %02d - Mean train loss = %.5f              ' %(epoch+1, avg_train_score/num_batch))
    print('        Test loss = %.5f' %(test_score))
    if arg.classification:
        acc = calcAccuracy(x1,y1)
        print('        Test accuracy = %.2f' %(acc), '%')
    print('')
    
    avg_time_per_epoch += (timer() - start)

    # Early stopping
    if arg.patience > 0: # When 0 or smaller, run until end of epochs
        if test_score > prev_test_score:
            early_stop_count += 1
            if early_stop_count == arg.patience:
                print('Early stopping condition reached')
                break 
        else:
            early_stop_count = 0
            saveModel(model, save_path)
            prev_test_score = test_score

avg_time_per_epoch = avg_time_per_epoch/arg.num_epoch
print('Average time per epoch: %.2fs' %avg_time_per_epoch)

# ---- REVERT TO BEST MODEL ----
if arg.patience > 0:
    print('Load best known configuration (test loss = %.5f)\n' %prev_test_score)    
    model.load_state_dict(torch.load('%s/PoNDeR.pth' % (save_path))) # Load best known configuration

# ---- PLOTTING ----

print('Running eval on train set', end='\r')
train_score,x2,y2 = evaluateModel(model, test_loss_func, dataloader, arg.dual, arg.CUDA, classification=arg.classification)
if arg.classification:
    acc = calcAccuracy(x2,y2)
print('Final train loss = %.5f, accuracy = %.2f' %(train_score, acc), '%')

if not arg.classification:
    print('Creating plot...')
    scatter(x1.data.cpu(), y1.data.cpu(), x2.data.cpu(), y2.data.cpu(), prev_test_score, save_path)