import torch
import torch.nn as nn
import torch.nn.functional as F

# Feature extraction and point cloud maxpooling
class PointNetFeat(nn.Module):
    def __init__(self, num_points = 2500, in_channels = 3):
        super(PointNetFeat, self).__init__()
        
        self.num_points = num_points
        
        self.conv1 = torch.nn.Conv1d(in_channels, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        self.mp1 = torch.nn.MaxPool1d(num_points)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)
        return x

# Feature to class mapping
class PointNetClass(nn.Module):
    def __init__(self, num_points = 2500, num_class = 2):
        super(PointNetClass, self).__init__()
        
        self.num_points = num_points
        self.feat = PointNetFeat(num_points)
        
        self.lin1 = nn.Linear(1024, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, num_class)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.do1 = nn.Dropout(0.3) # or batchnorm1d(256) ?
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.feat(x)
        x = F.relu(self.bn1(self.lin1(x)))
        x = F.relu(self.do1(self.lin2(x)))
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)