import torch
from torch.autograd import Variable
import torch.nn.functional as F


def evaluateModel(model, testloader):
    model.eval()  # Set to testing mode
    cnt = 0
    loss_sum = 0
    for data in testloader:
        points, target = data
        points = Variable(points)
        target = Variable(target)
        points = points.transpose(2, 1)
        prediction = model(points)
        loss = F.mse_loss(prediction, target, size_average=False)
        cnt += target.size(0)
        loss_sum += loss
    return loss_sum / cnt
