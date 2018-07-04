import torch
import torch.nn as nn
import torch.nn.functional as F

# Feature extraction and point cloud maxpooling

class PointNetFeat(nn.Module):
    def __init__(self, in_channels, num_points=250, avgPool=False):
        super(PointNetFeat, self).__init__()

        self.num_points = num_points

        self.conv1 = torch.nn.Conv1d(in_channels, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        if avgPool:
            self.pl = torch.nn.AvgPool1d(num_points)
        else:
            self.pl = torch.nn.MaxPool1d(num_points)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.pl(x)
        x = x.view(-1, 1024)
        return x

# Feature to value mapping

class PointNet(nn.Module):
    def __init__(self, in_channels, num_points=250, avgPool=False, sigmoid=False):
        super(PointNet, self).__init__()

        self.sigmoid = sigmoid
        self.num_points = num_points
        self.in_channels = in_channels
        self.feat = PointNetFeat(in_channels, num_points, avgPool)

        self.lin1 = nn.Linear(1024, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.do1 = nn.Dropout(0.3)  # or batchnorm1d(256) ?

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.feat(x)
        x = F.relu(self.bn1(self.lin1(x)))
        x = F.relu(self.do1(self.lin2(x)))
        x = self.lin3(x)
        if self.sigmoid:
            return nn.Sigmoid(x)
        else:
            return x

class DualPointNet(nn.Module):
    def __init__(self, in_channels, num_points=250, avgPool=False, sigmoid=False):
        super(DualPointNet, self).__init__()

        self.sigmoid = sigmoid
        self.num_points = num_points
        self.in_channels = in_channels
        self.feat = PointNetFeat(in_channels, num_points, avgPool)

        self.lin1 = nn.Linear(2048, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.do1 = nn.Dropout(0.3)  # or batchnorm1d(256) ?

        self.relu = nn.ReLU()

    def forward(self, ab):
        print(ab.shape)
        ab = torch.chunk(ab, 2, dim=0) # Split between proteins (see dataset where original concatenation happens)
        print(ab[0].shape)
        a = self.feat(ab[0])
        b = self.feat(ab[1])
        x = torch.cat((a,b), dim=0) # Concatenate both global features before passing to fully connected network
        print(x.shape)
        x = F.relu(self.bn1(self.lin1(x)))
        x = F.relu(self.do1(self.lin2(x)))
        x = self.lin3(x)
        if self.sigmoid:
            return nn.Sigmoid(x)
        else:
            return x