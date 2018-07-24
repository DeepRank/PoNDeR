import torch
import os

# Get current learning rate from optimizer
def get_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr

# Save model
def saveModel(model, path):
    torch.save(model.state_dict(), '%s/PoNDeR.pth' % (path))

# favorHighLoss
def favor_high_loss(input, target, size_average=True, reduce=True):
    d = torch.abs(((input - target)**2)*(target+1)) # High inputs biased in loss
    if not reduce:
        return d
    return torch.mean(d) if size_average else torch.sum(d)

class FavorHighLoss(torch.nn.Module):
    def __init__(self, size_average=True, reduce=True):
        super().__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, input, target):
        assert not target.requires_grad
        return favor_high_loss(input, target, size_average=self.size_average, reduce=self.reduce)

# Returns classification accuracy in percent
def calcAccuracy(x,y):
    x = x.cpu().data
    y = y.cpu().data
    _, max_indices = torch.max(y,1)
    acc = 100*(max_indices == x).sum()/len(x)
    return acc