import torch
from torch.autograd import Variable
import torch.nn.functional as F

def evaluateModel(model, loss_func, testloader, dual = False, CUDA = False):
    model.eval()  # Set to testing mode
    cnt = 0
    loss_sum = 0
    targets = []
    predictions = []
    for data in testloader:

        if dual:
            pointsA, pointsB, target = data
            pointsA, pointsB, target = Variable(pointsA, volatile=True), Variable(pointsB, volatile=True), Variable(target, volatile=True)  # Deprecated in PyTorch >=0.4
            pointsA, pointsB = pointsA.transpose(2,1), pointsB.transpose(2,1)
            if CUDA:
                pointsA, pointsB, target = pointsA.cuda(), pointsB.cuda(), target.cuda()
            prediction = model((pointsA, pointsB)).view(-1)
        else:
            points, target = data
            points, target = Variable(points, volatile=True), Variable(target, volatile=True)  # Deprecated in PyTorch >=0.4
            points = points.transpose(2, 1)
            if CUDA:
                points, target = points.cuda(), target.cuda()
            prediction = model(points).view(-1)

        loss = loss_func(prediction, target)

        cnt += target.size(0)
        loss_sum += loss.data[0]
        
        predictions.append(prediction)
        targets.append(target)
    return loss_sum / cnt, torch.cat(targets), torch.cat(predictions)