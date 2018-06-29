import torch

# Get current learning rate from optimizer
def get_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr

# Save model

def saveModel(model, arg):
    torch.save(model.state_dict(), '%s/PPIPointNet.pth' % (arg.out_folder))
