# package to define commonly used functions: plot curves, get results etc
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
    accuracy_score
)

def roc(prediction, truth, save_path=None):     
    plt.figure()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    fpr, tpr, thresholds = roc_curve(
                truth, prediction, drop_intermediate=False
            )
    gmeans = np.sqrt(tpr * (1 - fpr))
    # if abs(lim_min - 16.23)<0.1:
    # print(thresholds)
    # get the index of the maximum value of gmean
    ix = np.argmax(gmeans)
    print(
        "Best Threshold=%.3f, G-Mean=%.3f"
        % (thresholds[ix], gmeans[ix])
    )
    plt.plot(
        fpr, tpr
    )

    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best = %f' % thresholds[ix])
    plt.plot([0, 0, 1], [0, 1, 1],color="k", linestyle="--", label="ideal")
    plt.plot([0, 1], [0, 1],color="k", linestyle="-.", label = "no skill")

    plt.legend(loc="lower right")
    plt.grid(color="k", linestyle="--", linewidth=0.1)
    plt.title("ROC curve")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return thresholds[ix]

def f1_mcc(prediction, truth, save_path=None):
    """plot the MCC - F1 for a list of thresholds,
    in order to determine the best threshold"""
    plt.figure()
    plt.xlabel("F1 score")
    plt.ylabel("MCC score")
    thresholds = [0.01 * i for i in range(101)]
    mcc = []
    f1 = []
    for thres in thresholds:
        y_pred = [int(x > thres) for i, x in enumerate(prediction)]
        mcc.append(matthews_corrcoef(truth, y_pred))
        f1.append(f1_score(truth, y_pred))

    # the best threshold corresponds to the value where both F1 score and MCC are maximized
    # so we get the index of the maximum value of the sum (f1+MCC)
    T = np.asarray([f1[i] + mcc[i] for i in range(len(mcc))])
    best = np.argmax(T)
    # We plot the curve
    plt.plot(f1, mcc)
    # we draw the point corresponding to the best threshold
    plt.scatter(
        f1[best],
        mcc[best],
        marker="o",
        color="black",
        label="Best threshold = %.2f" % thresholds[best],
    )
    plt.scatter(0, 0, marker="o", color="r", label="worst case")
    plt.scatter(1, 1, marker="o", color="b", label="best case")
    plt.legend(loc="upper left")
    plt.grid(color="k", linestyle="--", linewidth=0.1)
    plt.title("MCC-F1 Curve")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return thresholds[best]

def probability_dist(prediction, save_path=None):
    plt.hist(prediction, density=True, bins=100)
    plt.title("probability distribution for class True")
    if save_path:
        plt.savefig(save_path)
    plt.show()
def history_epochs(history, save_path=None):
    _, axis = plt.subplots()
    axis.set_xlabel("epoch")
    axis.set_ylabel("performance")
    plt.plot(history.history["loss"], color="b", linestyle="--")
    plt.plot(history.history["val_loss"], color="r",linestyle="--")
    plt.grid(color="k", linestyle="--", linewidth=0.1)
    plt.plot(history.history["accuracy"], color="b")
    plt.plot(history.history["val_accuracy"], color="r")
    plt.title("model performance")
    plt.legend(["train", "test"], loc="center right")
    plt.show()
    if save_path:
        plt.savefig(save_path)
