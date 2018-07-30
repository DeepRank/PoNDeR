import os
import matplotlib
if os.environ.get('DISPLAY','') == '':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import h5py

'''
Creates a histogram of number of points per point cloud in the provided file
'''

hf = h5py.File('FINAL_Pairs.h5','r')

def getLengths(groupName):
    lengths = []
    group = hf[groupName]
    keys = list(group.keys())
    for key in keys:
#       FOR DUAL TYPE FILES
#       pcA = np.array(group.get(key).get('A'))
#       pcB = np.array(group.get(key).get('B'))
#       lengths.append(len(pcA)+len(pcB))
        pc = np.array(group.get(key))
        lengths.append(len(pc))
    return lengths

lengths = []
lengths.extend(getLengths('train'))
lengths.extend(getLengths('test'))
lengths.extend(getLengths('holdout'))

hf.close()

fig = plt.figure()
ax = fig.gca()

ax.set_ylabel('Amount')
ax.set_xlabel('Number of points')

ax.hist(lengths, bins=20)

fig.set_size_inches(5, 3.5)
fig.tight_layout()

fig.savefig('lenHist.png', dpi=100)