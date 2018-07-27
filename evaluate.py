import torch
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, matthews_corrcoef, f1_score

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

# Convert prediction to labels
def predToLabel(pred):
    _, pred_label = torch.max(pred,1)
    return pred_label

# Returns Matthew Correlation Coefficient
def calcMCC(truth, pred):
    pred_label = predToLabel(pred)
    return matthews_corrcoef(truth, pred_label)

# Returns F1 score
def calcF1(truth, pred):
    pred_label = predToLabel(pred)
    return f1_score(truth, pred_label)

# Returns confusion matrix
def calcConfusionMatrix(truth, pred):
    pred_label = predToLabel(pred)
    return confusion_matrix(truth, pred_label)