import numpy as np
import matplotlib.pyplot as plt   

def plotDiscriminant(true_seg, false_seg, nBins, filename):
    plt.hist(true_seg, nBins, color='blue', label='Real Segments', alpha=0.7)
    plt.hist(false_seg, nBins, color='red', label='Fake Segments', alpha=0.7)
    plt.xlim(0, 1)
    plt.xlabel("Edge Weight")
    plt.ylabel("Counts")
    plt.title("Edge Weights: All Segments")
    plt.legend(loc="upper right")
    plt.savefig(filename, dpi=1200)
    plt.clf()

def confusionMatrix(true_seg, false_seg, cut):
    true_seg, false_seg = np.array(true_seg), np.array(false_seg)
    Nt, Nf = len(true_seg), len(false_seg)
    TP = len(true_seg[true_seg > cut])/Nt    # true positive
    FN = len(true_seg[true_seg < cut])/Nt    # false negative
    FP = len(false_seg[false_seg > cut])/Nf  # false positive
    TN = len(false_seg[false_seg < cut])/Nf  # true negative
    return np.array([[TP, FN], [FP, TN]])

def confusionPlot(true_seg, false_seg, filename):
    cuts = np.arange(0, 1, 0.01)
    matrices = [confusionMatrix(true_seg, false_seg, i) for i in cuts]

    plt.scatter(cuts, [matrices[i][0][0] for i in range(len(cuts))], 
                label="True Positive", color='mediumseagreen', marker='h', s=1.8)
    plt.scatter(cuts, [matrices[i][0][1] for i in range(len(cuts))],
                label="False Negative", color='orange', marker='h', s=1.8)
    plt.scatter(cuts, [matrices[i][1][0] for i in range(len(cuts))], 
                label="False Positive", color='red', marker='h', s=1.8)
    plt.scatter(cuts, [matrices[i][1][1] for i in range(len(cuts))],
                label="True Negative", color='mediumslateblue', marker='h', s=1.8)
    plt.xlabel('Discriminant Cut')
    plt.ylabel('Yield')
    plt.legend(loc='best')
    plt.savefig(filename, dpi=1200)
    plt.clf()

def plotROC(true_seg, false_seg, filename):
    cuts = np.logspace(-6, 0, 1000)
    matrices = [confusionMatrix(true_seg, false_seg, i) for i in cuts]
    plt.scatter([matrices[i][1][0] for i in range(len(cuts))], 
                [matrices[i][0][0] for i in range(len(cuts))], 
                color='goldenrod', marker='h', s=2.2)
    plt.grid(True)
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(filename, dpi=1200)
    plt.clf()
