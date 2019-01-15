from matplotlib import pyplot as plt
import numpy as np


def save_auroc(auc, roc, filename):
    plt.plot(*zip(*roc), label='ROC curve')
    plt.plot([0, 1], label='Random guess', linestyle='--', color='red')
    plt.legend(loc=4)
    plt.ylabel('TPR (True Positive Rate)')
    plt.xlabel('FPR (False Positive Rate)')
    plt.title('ROC Curve (AUROC : %7.3f)' % (auc))
    plt.axis([0, 1, 0, 1])
    plt.grid()
    plt.savefig(filename)
    plt.clf()
    plt.close()


def compute_auroc(predict, target):
    n = len(predict)

    # Cutoffs are of prediction values
    cutoff = predict

    TPR = [0 for x in range(n + 2)]
    FPR = [0 for x in range(n + 2)]

    for k in range(n):
        predict_bin = 0

        TP = 0  # True Positive
        FP = 0  # False Positive
        FN = 0  # False Negative
        TN = 0  # True Negative

        for j in range(n):
            if (predict[j] >= cutoff[k]):
                predict_bin = 1
            else:
                predict_bin = 0

            #print(predict_bin, target[j])
            TP = TP + (predict_bin & target[j])
            FP = FP + (predict_bin & (not target[j]))
            FN = FN + ((not predict_bin) & target[j])
            TN = TN + ((not predict_bin) & (not target[j]))

        # value to prevent divison by zero
        # Implement It is divided into smaller values instead, If you divide by 0 when you compute TPR, FPR
        very_small_value = 1e-10

        # True Positive Rate
        # TPR[k] = float(TP) / float(TP + FN)
        TPR[k] = float(TP) / (float(TP + FN) if float(TP + FN) != 0 else very_small_value)
        # False Positive Rate
        # FPR[k] = float(FP) / float(FP + TN)
        FPR[k] = float(FP) / (float(FP + TN) if float(FP + TN) != 0 else very_small_value)

    TPR[n] = 0.0
    FPR[n] = 0.0
    TPR[n + 1] = 1.0
    FPR[n + 1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, TPR), reverse=True)

    AUROC = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n + 1):
        h = ROC[j][0] - ROC[j + 1][0]
        w = (ROC[j][1] + ROC[j + 1][1]) / 2.0

        AUROC = AUROC + h * w

    return AUROC, ROC

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
