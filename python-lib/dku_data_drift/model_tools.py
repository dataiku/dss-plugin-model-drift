# coding: utf-8
import logging
import numpy as np
from sklearn.metrics import *

logger = logging.getLogger(__name__)


def mroc_auc_score(y_true, y_predictions, sample_weight=None):
    """ Returns a auc score. Handles multi-class
    For multi-class, the AUC score is in fact the MAUC
    score described in
    David J. Hand and Robert J. Till. 2001.
    A Simple Generalisation of the Area Under the ROC Curve
    for Multiple Class Classification Problems.
    Mach. Learn. 45, 2 (October 2001), 171-186.
    DOI=10.1023/A:1010920819831
    http://dx.doi.org/10.1023/A:1010920819831
    """
    (nb_rows, max_nb_classes) = y_predictions.shape
    # Today, it may happen that if a class appears only once in a dataset
    # it can appear in the train and not in the validation set.
    # In this case it will not be in y_true and
    # y_predictions.nb_cols is not exactly the number of class
    # to consider when computing the mroc_auc_score.
    classes = np.unique(y_true)
    nb_classes = len(classes)
    if nb_classes > max_nb_classes:
        raise ValueError("Your test set contained more classes than the test set. Check your dataset or try a different split.")

    if nb_classes < 2:
        raise ValueError("Ended up with less than two-classes in the validation set.")

    if nb_classes == 2:
        classes = classes.tolist()
        y_true = y_true.map(lambda c: classes.index(c)) # ensure classes are [0 1]
        return roc_auc_score(y_true, y_predictions[:, 1], sample_weight=sample_weight)

    def A(i, j):
        """
        Returns a asymmetric proximity metric, written A(i | j)
        in the paper.
        The sum of all (i, j) with  i != j
        will give us the symmetry.
        """
        mask = np.in1d(y_true, np.array([i, j]))
        y_true_i = y_true[mask] == i
        y_pred_i = y_predictions[mask][:, i]
        if sample_weight is not None:
            sample_weight_i = sample_weight[mask]
        else:
            sample_weight_i = None
        return roc_auc_score(y_true_i, y_pred_i, sample_weight=sample_weight_i)

    C = 1.0 / (nb_classes * (nb_classes - 1))
    # TODO: double check
    return C * sum(
        A(i, j)
        for i in classes
        for j in classes
        if i != j)