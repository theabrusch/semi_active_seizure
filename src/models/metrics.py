import numpy as np
import sklearn


def sensitivity(y_true, y_pred):
    TP = ((y_true == 1) & (y_pred == 1)).sum()
    FN = ((y_true == 1) & (y_pred == 0)).sum()

    sens = TP/(TP+FN)

    return sens

def specificity(y_true, y_pred):
    TN = ((y_true == 0) & (y_pred == 0)).sum()
    FP = ((y_true == 0) & (y_pred == 1)).sum()

    spec = TN/(TN+FP)

    return spec

def accuracy(y_true, y_pred):
    correct = (y_true==y_pred).sum()
    acc = correct/len(y_true)
    return acc

def get_metrics(metric_names):
    metrics = []
    for metric in metric_names:
        if metric == 'sensitivity':
            metrics.append(sensitivity)
        elif metric == 'specificity':
            metrics.append(specificity)
        elif metrics == 'accuracy':
            metrics.append(accuracy)
        else:
            print('Metric', metric, 'not implemented.')

    return metrics
