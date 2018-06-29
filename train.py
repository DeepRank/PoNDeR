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
from evaluate import evaluateModel
from dataset import PDBset
from utils import get_lr

# PRINT INFORMATION 

print('ABOUT')
print('    Simplified PointNet for Protein-Protein Reaction - Training script')
print('    Lukas De Clercq, 2018, Netherlands eScience Center\n')

print('RUNTIME INFORMATION')
print('    System    -', platform.system(), platform.release(), platform.machine())
print('    Version   -', platform.version())
print('    Node      -', platform.node())
print('    Time      -', datetime.datetime.utcnow(), 'UTC', '\n')

print('LIBRARY VERSIONS')
print('    Python    -', platform.python_version(),'on', platform.python_compiler())
print('    Pytorch   -', torch.__version__)
print('    CUDA      -', torch.version.cuda)
print('    CUDNN     -', torch.backends.cudnn.version(), '\n')

# ---- OPTION PARSING ----

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int,  default=50,   help='Input batch size')
parser.add_argument('--num_points', type=int,  default=350, help='Points per point cloud used')
parser.add_argument('--num_workers',type=int,  default=4,    help='Number of data loading workers')
parser.add_argument('--num_epoch',  type=int,  default=5,   help='Number of epochs to train for')
parser.add_argument('--cosine_decay', dest='cosine_decay', default=False, action='store_true', help='Use cosine annealing for learning rate decay')
parser.add_argument('--CUDA', dest='CUDA', default=False, action='store_true', help='Train on GPU')
parser.add_argument('--out_folder', type=str,  default='/artifacts',  help='Model output folder')
parser.add_argument('--model',      type=str,  default='',   help='Model input path')
parser.add_argument('--data_path', type=str, default='/home/lukas/DR_DATA/pointclouds/')

arg = parser.parse_args()
print('RUN PARAMETERS')
print('    ', arg, '\n')

# ---- DATA LOADING ----

dataset = PDBset(train = True, num_points = arg.num_points, root_dir=arg.data_path)
dataloader = data.DataLoader(dataset,batch_size=arg.batch_size,shuffle=True,num_workers=int(arg.num_workers))

testset = PDBset(train = False, num_points = arg.num_points, root_dir=arg.data_path)
testloader = data.DataLoader(testset,batch_size=arg.batch_size,shuffle=True,num_workers=int(arg.num_workers))

num_batch = len(dataset)/arg.batch_size

print('DATA PARAMETERS')
print('    Set sizes: %d & %d -> %.1f' % (len(testset), len(dataset), 100*len(testset)/len(dataset)), '%')

# ---- SET UP MODEL ----

print('MODEL PARAMETERS')
model = PointNet(num_points = arg.num_points, in_channels = 8)
if arg.model != '': model.load_state_dict(torch.load(arg.model))
if arg.CUDA: model.cuda()
print(model)

optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_batch+1)

# ---- INITIAL TEST SET EVALUATION ----

print('START EVALUATION OF RANDOM WEIGHTS')
pretrain_test_score = evaluateModel(model, testloader)
print('    Pre-train test score =', pretrain_test_score)

# ---- MODEL TRAINING ----

print('START TRAINING')
model.train() # Set to training mode

for epoch in range(arg.num_epoch):
    
    scheduler.base_lrs = [0.002*(1-(epoch**2)/(arg.num_epoch**2))]
    scheduler.step(epoch=0)
    
    for i, data in enumerate(dataloader, 0):
        points, target = data
        points, target = Variable(points), Variable(target) # Deprecated in PyTorch >=0.4
        points = points.transpose(2,1)
        if arg.CUDA: points, target = points.cuda(), target.cuda()
        
        optimizer.zero_grad()
        prediction = model(points)
        loss = F.mse_loss(prediction, target)
        loss.backward()
        
        optimizer.step()
        if arg.cosine_decay:
            scheduler.step()
        
        print('    e%d - %d/%d - LR: %f - Loss: %.3f' %(epoch, i, num_batch, get_lr(optimizer)[0], loss))

    print('')

# ---- SAVE MODEL ----

print('SAVING MODEL')
def saveModel(model, arg):
    torch.save(model.state_dict(), '%s/PPIPointNet.pth' % (arg.out_folder))

# ---- FINAL TEST SET EVALUATION ----

print('START EVALUATION')
posttrain_test_score = evaluateModel(model, testloader)
print('    Post-train test score =', posttrain_test_score)
