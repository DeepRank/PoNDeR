import torch
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, matthews_corrcoef

def evaluateModel(model, loss_func, testloader, dual=False, CUDA=False, classification=False):
    model.eval()  # Set to testing mode
    cnt = 0
    loss_sum = 0
    targets = []
    predictions = []
    for data in testloader:
        points, target = data
        points, target = Variable(points, volatile=True), Variable(target, volatile=True)  # Deprecated in PyTorch >=0.4
        points = points.transpose(2, 1)
        if CUDA:
            points, target = points.cuda(), target.cuda()
        prediction = model(points)
        if not classification:
            prediction = prediction.view(-1)
        loss = loss_func(prediction, target)
        cnt += target.size(0)
        loss_sum += loss.data[0]
        predictions.append(prediction)
        targets.append(target)
    model.train()
    return loss_sum / cnt, torch.cat(targets), torch.cat(predictions)

# Returns Matthew Correlation Coefficient
def calcMCC(truth, pred):
    _, predLabel = torch.max(pred,1) # To label
    return matthews_corrcoef(truth, predLabel)

# Returns confusion matrix
def calcConfusionMatrix(truth, pred):
    _, max_indices = torch.max(pred,1)
    return confusion_matrix(truth, max_indices)