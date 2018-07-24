import os
import matplotlib
if os.environ.get('DISPLAY','') == '':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.axes as axs
import seaborn as sn
import pandas as pd

def plotScatter(x1, y1, x2, y2, test_score, save_path):
    fig, ax = plt.subplots()
    ax.scatter(x2,y2, label='Train',s=1)
    ax.scatter(x1,y1, label='Test' ,s=1)
    ax.set_ylabel('Prediction')
    ax.set_xlabel('Truth')
    ax.set_xlim(xmin=0.0) # All scores are > 0
    ax.set_ylim(ymin=0.0)
    ax.legend(loc='best')
    title = 'Test loss: %.5f' %test_score # Best known test score
    fig.suptitle(title)
    fig.set_size_inches(5, 5)
    figname = save_path + '/scatter.png'
    fig.savefig(figname, dpi=100)

def plotConfusionMatrix(matrix, save_path):
    plt.figure(figsize = (5,5))
    sn.heatmap(matrix, annot=True)
    figname = save_path + '/matrix.png'
    plt.savefig(figname, dpi=100)