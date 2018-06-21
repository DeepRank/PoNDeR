import os
import platform
import datetime
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable

import numpy as np

from PPIPointNet import PointNet
from dataset import PDB
from utils import get_lr

# PRINT INFORMATION 

print('ABOUT')
print('  Simplified PointNet for Protein-Protein Reaction - Training script')
print('  Lukas De Clercq, 2018, Netherlands eScience Center\n')

print('RUNTIME INFORMATION')
print('  System    -', platform.system(), platform.release(), platform.machine())
print('  Version   -', platform.version())
print('  Node      -', platform.node())
print('  Time      -', datetime.datetime.utcnow(), 'UTC', '\n')

print('LIBRARY VERSIONS')
print('  Python    -', platform.python_version(),'on', platform.python_compiler())
print('  Pytorch   -', torch.__version__)
print('  CUDA      -', torch.version.cuda)
print('  CUDNN     -', torch.backends.cudnn.version(), '\n')

# ---- OPTION PARSING ----

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int,  default=32,   help='Input batch size')
parser.add_argument('--num_points', type=int,  default=2500, help='Points per point cloud used')
parser.add_argument('--num_workers',type=int,  default=4,    help='Number of data loading workers')
parser.add_argument('--num_epoch',  type=int,  default=25,   help='Number of epochs to train for')
parser.add_argument('--cosine_decay', dest='cosine_decay', default=False, action='store_true', help='Use cosine annealing for learning rate decay')
parser.add_argument('--epoch_decay', dest='epoch_decay', default=False, action='store_true', help='Decay learning rate per epoch')
parser.add_argument('--CUDA', dest='CUDA', default=False, action='store_true', help='Train on GPU')
parser.add_argument('--out_folder', type=str,  default='/artifacts',  help='Model output folder')
parser.add_argument('--model',      type=str,  default='',   help='Model input path')

arg = parser.parse_args(['--num_epoch','3','--cosine_decay', '--epoch_decay'])
print('RUN PARAMETERS')
print('  ', arg, '\n')

# ---- DATA LOADING ----

dataset = PDB(train = True, num_points = arg.num_points)
dataloader = data.DataLoader(dataset,batch_size=arg.batch_size,shuffle=True,num_workers=int(arg.num_workers))

testset = PDB(train = False, num_points = arg.num_points)
testloader = data.DataLoader(testset,batch_size=arg.batch_size,shuffle=True,num_workers=int(arg.num_workers))

num_batch = len(dataset)/arg.batch_size

print('DATA PARAMETERS')
print('  Set sizes: %d & %d -> %.1f' % (len(testset), len(dataset), 100*len(testset)/len(dataset)), '%')

# ---- SET UP MODEL ----

print('MODEL PARAMETERS')
model = PointNet(num_points = arg.num_points)
if arg.model != '': model.load_state_dict(torch.load(arg.model))
if arg.CUDA: model.cuda()
print(model)

optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_batch)

#lossProg = np.zeros(len(dataloader)*arg.num_epoch)
#learnProg = np.zeros(len(dataloader)*arg.num_epoch)

# ---- MODEL TRAINING ----

print('START TRAINING')
model.train()

for epoch in range(arg.num_epoch):
    
    if arg.epoch_decay:
        scheduler.base_lrs = [0.02*(1-(epoch**2)/(arg.num_epoch**2))]
    
    for i, data in enumerate(dataloader, 0):
        points, goal = data
        points, goal = Variable(points), Variable(goal[:,0])
        points = points.transpose(2,1)
        #points, goal = points.cuda(), goal.cuda()
        
        optimizer.zero_grad()
        prediction = model(points)
        loss = F.mse_loss(prediction, goal)
        loss.backward()
        
        optimizer.step()
        if arg.cosine_decay:
            scheduler.step() # Decay within batch
        
        print('  e%d - %d/%d - LR: %f - Loss: %.3f' %(epoch, i, num_batch, get_lr(optimizer)[0], loss))

        # For plotting
        #lossProg[i+epoch*len(dataloader)]=loss.data[0]
        #learnProg[i+epoch*len(dataloader)]=get_lr(optimizer)[0]

# ---- EVALUATE ON TEST SET ----

model.eval()

correct = 0
total = 0
for data in testloader:
    points, target = data
    points = Variable(points)
    points = points.transpose(2,1)
    outputs = model(points)
    pred = torch.max(outputs.data, 1)
    total += target.size(0)
    correct += (pred[1] == target.transpose(0,1)).sum()
print('Accuracy on test set: %d %%' %(100 * correct / total))

model.train()

# ---- SAVE MODEL ----

torch.save(model.state_dict(), '%s/PPIPointNet.pth' % (arg.out_folder))