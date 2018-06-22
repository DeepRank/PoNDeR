decoy_fn = '1AK4_100w.pdb'
print(decoy_fn[:-9] + '.pdb')

# Get current learning rate from optimizer
def get_lr(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[param_group['lr']]
    return lr

# Get filename for native conformation from filename for simulated conformation
def get_ref(decoy_fn):
    ref_fn = decoy_fn.rsplit( "_", 1 )[ 0 ] + '.pdb'
    return ref_fn 