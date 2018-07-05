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

print('ABOUT')
print('    Simplified PointNet for Protein-Protein Reaction - Training script')
print('    Lukas De Clercq, 2018, Netherlands eScience Center\n')
print('    See attached license')

print('RUNTIME INFORMATION')
print('    System    -', platform.system(), platform.release(), platform.machine())
print('    Version   -', platform.version())
print('    Node      -', platform.node())
print('    Time      -', datetime.datetime.utcnow(), 'UTC', '\n')

print('LIBRARY VERSIONS')
print('    Python    -', platform.python_version(), 'on', platform.python_compiler())
print('    Pytorch   -', torch.__version__)
print('    CUDA      -', torch.version.cuda)
print('    CUDNN     -', torch.backends.cudnn.version(), '\n', flush = True)

# ---- OPTION PARSING ----

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=50, help='Input batch size')
parser.add_argument('--num_points', type=int, default=350, help='Points per point cloud used')
parser.add_argument('--num_workers', type=int,  default=1, help='Number of data loading workers')
parser.add_argument('--num_epoch',  type=int,  default=5, help='Number of epochs to train for')
parser.add_argument('--cosine_decay', dest='cosine_decay', default=False, action='store_true', help='Use cosine annealing for learning rate decay')
parser.add_argument('--CUDA', dest='CUDA', default=False, action='store_true', help='Train on GPU')
parser.add_argument('--out_folder', type=str, default='/artifacts',  help='Model output folder')
parser.add_argument('--model',      type=str, default='',   help='Model input path')
parser.add_argument('--data_path', type=str, default='/home/lukas/DR_DATA/pointclouds/')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--avg_pool', dest='avg_pool', default=False, action='store_true', help='Use average pooling after for feature pooling (instead of max pooling)')
parser.add_argument('--dual', dest='dual', default=False, action='store_true', help='Use DualPointNet architecture')
parser.add_argument('--get_min', dest='get_min', default=False, action='store_true', help='Get minimum point cloud size')

arg = parser.parse_args()
print('RUN PARAMETERS')
print('    ', arg, '\n', flush=True)

# ---- DATA LOADING ----

if arg.dual:
    dataset = DualPDBset(hdf5_file=arg.data_path, group='train', num_points=arg.num_points)
    testset = DualPDBset(hdf5_file=arg.data_path, group='test', num_points=arg.num_points)
else:
    dataset = PDBset(hdf5_file=arg.data_path, group='train', num_points=arg.num_points)
    testset = PDBset(hdf5_file=arg.data_path, group='test', num_points=arg.num_points)

dataloader = data.DataLoader(dataset, batch_size=arg.batch_size, shuffle=True, num_workers=int(arg.num_workers))
testloader = data.DataLoader(testset, batch_size=arg.batch_size, shuffle=True, num_workers=int(arg.num_workers))

num_batch = len(dataset)/arg.batch_size

print('DATA PARAMETERS')
print('    Test & train sizes: %d & %d -> %.1f' %(len(testset), len(dataset), 100*len(testset)/len(dataset)), '%', flush=True)

# ---- GET MINIMUM

if arg.get_min:
    minSize = min(dataset.getMin(), testset.getMin())
    print('    Minimum pointcloud size:', minSize)
print('')

# ---- SET UP MODEL ----

print('MODEL PARAMETERS')
if arg.dual:
    model = DualPointNet(num_points=arg.num_points, in_channels=6, avgPool=arg.avg_pool, sigmoid = True)
else:
    model = PointNet(num_points=arg.num_points, in_channels=6, avgPool=arg.avg_pool, sigmoid = True)

if arg.model != '':
    model.load_state_dict(torch.load(arg.model))

if arg.CUDA:
    model.cuda()
print(model)
print('')

optimizer = optim.SGD(model.parameters(), lr=arg.lr, momentum=0.9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_batch)
train_loss_func = FavorHighLoss()
test_loss_func = FavorHighLoss(size_average=False)

# ---- INITIAL TEST SET EVALUATION ----

print('START EVALUATION OF RANDOM WEIGHTS')
pretrain_test_score, _, _ = evaluateModel(model, test_loss_func, testloader, arg.dual, arg.CUDA)
print('    Pre-train test score = %.5f\n' %(pretrain_test_score))

# ---- MODEL TRAINING ----

print('START TRAINING')
model.train()  # Set to training mode

for epoch in range(arg.num_epoch):

    scheduler.base_lrs = [arg.lr*(1-(epoch**2)/(arg.num_epoch**2))]
    scheduler.step(epoch=0)

    for i, data in enumerate(dataloader, 0):
        optimizer.zero_grad()
        points, target = data
        points, target = Variable(points), Variable(target)  # Deprecated in PyTorch >=0.4
        if len(target) != arg.batch_size:
            break
        points = points.transpose(2, 1)
        if arg.CUDA:
            points, target = points.cuda(), target.cuda()
        prediction = model(points).view(-1)
        loss = train_loss_func(prediction, target)
        loss.backward()
        print('    E: %02d - %02d/%02d - LR: %.5f - Loss: %.5f' %(epoch+1, i+1, num_batch, get_lr(optimizer)[0], loss), flush=True)
        optimizer.step()
        if arg.cosine_decay:
            scheduler.step()
    print('')

# ---- SAVE MODEL ----

print('SAVING MODEL')
saveModel(model,arg)
print('    Model saved\n')

# ---- FINAL TEST SET EVALUATION ----

print('START EVALUATION')

posttrain_test_score,x1,y1 = evaluateModel(model, test_loss_func, testloader, arg.dual, arg.CUDA)

print('    Post-train test score = %.5f' %(pretrain_test_score))
print('    Improvement: %.1f %%\n' %(100*(pretrain_test_score-posttrain_test_score)/pretrain_test_score))

posttrain_train_score,x2,y2 = evaluateModel(model, test_loss_func, dataloader, arg.dual, arg.CUDA)

plt.scatter(x2,y2, label='Train',s=1)
plt.scatter(x1,y1, label='Test',s=1)
plt.xlim(0,1)
plt.ylim(0,1)
plt.title('Siamese Pointnet')
plt.xlabel('Truth')
plt.ylabel('Prediction')
plt.legend(loc=4)
plt.draw
plt.savefig('post-train.png')