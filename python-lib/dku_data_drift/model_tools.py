#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import numpy as np
import math
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from dku_data_drift.preprocessing import Preprocessor

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


def format_proba_density(data, sample_weight=None):
    data = np.array(data)
    if len(data) == 0:
        return []
    h = 1.06 * np.std(data) * math.pow(len(data), -.2)
    if h <= 0:
        h = 0.06
    if len(np.unique(data)) == 1:
        sample_weight = None
    X_plot = np.linspace(0, 100, 500, dtype=int)[:, np.newaxis]
    kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(data.reshape(-1, 1), sample_weight=sample_weight)
    Y_plot = [v if not np.isnan(v) else 0 for v in np.exp(kde.score_samples(X_plot))]
    return list(zip(X_plot.ravel(), Y_plot))


class SurrogateModel:

    def __init__(self, prediction_type):
        self.feature_names = None
        self.target = None
        self.prediction_type = prediction_type
        #TODO should we define some params of RF to avoid long computation ?
        if prediction_type == 'CLASSIFICATION':
            self.clf = RandomForestClassifier()
        else:
            self.clf = RandomForestRegressor()
        self.check()

    def check(self):
        if self.prediction_type not in ['CLASSIFICATION', 'REGRESSION']:
            raise ValueError('Prediction type must either be CLASSIFICATION or REGRESSION.')

    def fit(self, df, target):
        preprocessor = Preprocessor(df, target)
        train, test = preprocessor.get_processed_train_test()
        train_X = train.drop(target, axis=1)
        train_Y = train[target]
        self.clf.fit(train_X, train_Y)
        self.feature_names = train_X.columns

    def get_feature_importance(self, cumulative_percentage_threshold=95):
        feature_importance = []
        feature_importances = self.clf.feature_importances_
        for feature_name, feat_importance in zip(self.feature_names, feature_importances):
            feature_importance.append({
                'feature': feature_name,
                'importance': 100 * feat_importance / sum(feature_importances)
            })

        dfx = pd.DataFrame(feature_importance).sort_values(by='importance', ascending=False).reset_index(drop=True)
        dfx['cumulative_importance'] = dfx['importance'].cumsum()
        dfx_top = dfx.loc[dfx['cumulative_importance'] <= cumulative_percentage_threshold]
        return dfx_top.rename_axis('rank').reset_index().set_index('feature')