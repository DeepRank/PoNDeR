import torch
from torch.autograd import Variable
import torch.nn.functional as F


def evaluateModel(model, loss_func, testloader):
    model.eval()  # Set to testing mode
    cnt = 0
    loss_sum = 0
    for data in testloader:
        points, target = data
        points = Variable(points,volatile=True)
        target = Variable(target,volatile=True)
        points = points.transpose(2, 1)
        prediction = model(points)
        loss = loss_func(prediction, target)
        cnt += target.size(0)
        loss_sum += loss.data[0]
    return loss_sum / cnt
