import os
import matplotlib
if os.environ.get('DISPLAY','') == '':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.axes as axs

def scatter(x1, y1, x2, y2, test_score, save_path):
    fig, ax = plt.subplots()
    ax.scatter(x2.data.cpu(),y2.data.cpu(), label='Train',s=1)
    ax.scatter(x1.data.cpu(),y1.data.cpu(), label='Test' ,s=1)
    ax.set_ylabel('Prediction')
    ax.set_xlabel('Truth')
    ax.set_xlim(xmin=0.0) # All scores are > 0
    ax.set_ylim(ymin=0.0)
    ax.legend(loc='best')
    title = 'Test loss: %.5f' %test_score # Best known test score
    fig.suptitle(title)
    fig.set_size_inches(16, 9)
    figname = save_path + '/post-train.png'
    fig.savefig(figname, dpi=25)