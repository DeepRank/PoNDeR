import os
import matplotlib
if os.environ.get('DISPLAY','') == '':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.axes as axs
import seaborn as sn
import pandas as pd

'''
Plotting functions for model inspection
'''

# Create scatterplot (for regression)
def plotScatter(x1, y1, x2, y2, save_path, limit=False):
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 5)
    if limit:
        upper_bound = 1.0
    else:
        upper_bound = 50.0
    # All scores are > 0
    ax.set_xlim(xmin=0.0, xmax=upper_bound) 
    ax.set_ylim(ymin=0.0, ymax=upper_bound)
    ax.scatter(x2,y2, label='Train',s=1)
    ax.scatter(x1,y1, label='Test' ,s=1)
    ax.set_ylabel('Prediction')
    ax.set_xlabel('Truth')
    ax.legend(loc='upper left')
    figname = save_path + '/scatter.png'
    fig.savefig(figname, dpi=100)

# Create confusion matrix (for classification)
def plotConfusionMatrix(matrix, save_path):
    plt.figure(figsize = (5,5))
    sn.heatmap(matrix, annot=True, fmt='d', cmap='gist_heat', xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'], vmin=0)
    plt.xlabel('Prediction')
    plt.ylabel('Truth')
    figname = save_path + '/matrix.png'
    plt.savefig(figname, dpi=100)