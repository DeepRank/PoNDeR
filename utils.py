# Get current learning rate from optimizer
def get_lr(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[param_group['lr']]
    return lr