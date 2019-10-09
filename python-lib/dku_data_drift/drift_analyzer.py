# -*- coding: utf-8 -*-
import sys
import os
import json
import logging
import math

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower
from sklearn.neighbors import KernelDensity
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier 
from dataiku.doctor.prediction.dku_xgboost import DkuXGBClassifier # good idea ?
from preprocessing import  Preprocessor
from model_accessor import ModelAccessor
from model_tools import mroc_auc_score

logger = logging.getLogger(__name__)

ORIGIN_COLUMN = '__dku_row_origin__' # name for the column that will contain the information from where the row is from (original test dataset or new dataframe)
FROM_ORIGINAL = 'original'
FROM_NEW = 'new'
ACCEPTED_ALGORITHMS = [RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, DecisionTreeClassifier, DkuXGBClassifier]


class DriftAnalyzer:

    def __init__(self, model_accessor):
        self._model_accessor = model_accessor
        self._original_test_df = model_accessor.get_original_test_df()
        self._test_X = None
        self._test_Y = None
        self.check()

    def check(self):
        clf = self._model_accessor.get_predictor()._clf
        found_algorithm = False
        for algorithm in ACCEPTED_ALGORITHMS:
            if isinstance(clf, algorithm):
                found = True
                break
        if not found_algorithm:
            raise ValueError('{} is not a supported algorithm. Please choose one that has feature importances (tree-based models).'.format(clf.__module__))

    def train_drift_model(self, new_df):
        """
        Trains a classifier that attempts to discriminate between rows from the provided dataframe and
        rows from the dataset originally used to evaluate the model

        Returns (columns, classifier)
        """
        logger.info("Preparing the drift model...")
        df = self._prepare_data_for_drift_model(new_df)
        preprocessor = Preprocessor(df, target=ORIGIN_COLUMN)
        train, test = preprocessor.get_processed_train_test()

        train_X = train.drop(ORIGIN_COLUMN, axis=1)
        train_Y = np.array(train[ORIGIN_COLUMN])
        self._test_X = test.drop(ORIGIN_COLUMN, axis=1) # we will use them later when compute metrics
        self._test_Y = np.array(test[ORIGIN_COLUMN])

        clf = RandomForestClassifier(n_estimators=100, random_state=1337, max_depth=13, min_samples_leaf=1)
        logger.info("Fitting the drift model...")
        clf.fit(train_X, train_Y)
        return train_X.columns, clf

    def compute_drift_metrics(self, new_df, drift_features, drift_clf):
        logger.info("Computing drift metrics ...")
        feature_importance_metrics = self._get_feature_importance_metrics(drift_features, drift_clf, top_n=50)
        drift_auc = self._get_drift_auc(drift_clf)
        drift_accuracy = self._get_drift_accuracy(drift_clf)
        prediction_dict = self._get_predictions(new_df, limit=10000)

        predictions_by_class = {}
        for label in prediction_dict.get(FROM_ORIGINAL).columns:
            if 'proba_' in label:
                original_proba = np.around(prediction_dict.get(FROM_ORIGINAL)[label].values, 2).tolist()
                new_proba = np.around(prediction_dict.get(FROM_NEW)[label].values, 2).tolist()
                predictions_by_class[label] = {FROM_ORIGINAL: original_proba, FROM_NEW: new_proba}
        kde_dict = self._get_kde(predictions_by_class)
        fugacity_metrics = self._get_fugacity(prediction_dict)
        stat_metrics = self._get_stat_test3(kde_dict)
        label_list = [label for label in fugacity_metrics[0].keys() if label != 'source']
        return {'feature_importance': feature_importance_metrics, 'drift_accuracy': drift_accuracy, 'kde': kde_dict, 'stat_metrics':stat_metrics, 'fugacity': fugacity_metrics, 'label_list': label_list}

    def _prepare_data_for_drift_model(self, new_test_df):
        target = self._model_accessor.get_target_variable()
        original_df = self._original_test_df.drop(target, axis=1)
        if target in new_test_df:
            new_df = new_test_df.drop(target, axis=1)
        else:
            new_df = new_test_df.copy()

        original_df[ORIGIN_COLUMN] = FROM_ORIGINAL
        new_df[ORIGIN_COLUMN] = FROM_NEW

        logger.info("Rebalancing data:")
        number_of_rows = min(original_df.shape[0], new_df.shape[0])
        logger.info(" - original test dataset had %s rows, new dataframe has %s. Selecting %s for each." % (original_df.shape[0], new_df.shape[0], number_of_rows))

        df = pd.concat([original_df.head(number_of_rows), new_df.head(number_of_rows)], sort=False)
        selected_features = [ORIGIN_COLUMN] + self._model_accessor.get_selected_features()
        return df.loc[:, selected_features]

    def _get_kde(self, predictions):
        kde_dict = {}
        for label in predictions.keys():
            kde_original = self._format_proba_density(predictions.get(label).get(FROM_ORIGINAL))
            kde_new = self._format_proba_density(predictions.get(label).get(FROM_NEW))
            kde_dict[label] = {FROM_ORIGINAL: kde_original, FROM_NEW: kde_new}
        return kde_dict

    def _format_proba_density(self, data, sample_weight=None):
        data = np.array(data)
        if len(data) == 0:
            return []
        h = 1.06 * np.std(data) * math.pow(len(data), -.2)
        if h <= 0:
            h = 0.06
        if len(np.unique(data)) == 1:
            sample_weight = None
        X_plot = np.linspace(0, 100, 500, dtype='int')[:, np.newaxis]
        kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(data.reshape(-1, 1), sample_weight=sample_weight)
        Y_plot = np.exp(kde.score_samples(X_plot))
        Y_plot = [v if not np.isnan(v) else 0 for v in np.exp(kde.score_samples(X_plot))]
        return zip(X_plot.ravel(), Y_plot)

    def _get_drift_feature_importance(self, drift_features, drift_clf):
        feature_importance = []
        for feature_name, feat_importance in zip(drift_features, drift_clf.feature_importances_):
            feature_importance.append({
                'feature': feature_name,
                'importance': 100*feat_importance/sum(drift_clf.feature_importances_)
            })

        dfx = pd.DataFrame(feature_importance).sort_values(by='importance', ascending=False).reset_index(drop=True)#.drop('importance', axis=1)
        return dfx.rename_axis('rank').reset_index().set_index('feature')

    def _get_stat_test3(self, kde_dict, alpha=0.05):
        power_analysis = TTestIndPower()
        stat_test_dict = {}
        for label in kde_dict.keys():
            kde_original = [x[1] for x in kde_dict.get(label).get(FROM_ORIGINAL)]
            kde_new = [x [1] for x in kde_dict.get(label).get(FROM_NEW)]
            # this effect size has an equal variance assumption (?)
            effect_size = np.abs((np.mean(kde_original) - np.mean(kde_new)))/np.std(kde_original)
            power = power_analysis.power(effect_size=effect_size, nobs1=len(kde_original), alpha=0.05)
            t_test = stats.ttest_ind(kde_original, kde_new, equal_var=False)[-1]
            stat_test_dict[label] = {'t_test': round(t_test,4), 'power': round(power,4)}

        return stat_test_dict

    def _get_feature_importance_metrics(self, drift_features, drift_clf, top_n):
        original_feature_importance_df = self._model_accessor.get_feature_importance()
        drift_feature_importance_df = self._get_drift_feature_importance(drift_features, drift_clf)
        topn_drift_feature = drift_feature_importance_df[:top_n].to_dict()['importance']
        topn_original_feature = original_feature_importance_df.loc[topn_drift_feature.keys()].to_dict()['importance']
        feature_importance_metrics = []
        for feature in topn_original_feature.keys():
            drift_feat_rank = topn_drift_feature.get(feature)
            if np.isnan(drift_feat_rank):
                logger.warn('Feature {} does not exist in the orignal test set.'.format(feature))
            elif np.isnan(topn_original_feature.get(feature, 0)):
                logger.warn('No original_model.') #TODO no idea what this means, did not think, just added this list to avoid nan here
            else:
                feature_importance_metrics.append({'original_model': topn_original_feature.get(feature, 0), 'drift_model': drift_feat_rank, 'feature': feature})
        return feature_importance_metrics

    def _get_drift_auc(self, drift_clf):
        probas = drift_clf.predict_proba(self._test_X)
        test_Y_ser = pd.Series(self._test_Y)
        auc_score = mroc_auc_score(test_Y_ser, probas)
        return auc_score

    def _get_drift_accuracy(self, drift_clf):
        predicted_Y = drift_clf.predict(self._test_X)
        test_Y = pd.Series(self._test_Y)
        drift_accuracy = round(accuracy_score(test_Y, predicted_Y),2)
        return drift_accuracy

    def _get_predictions(self, new_df, limit=10000):
        """
        The result of model_accessor.predict() is a dataframe prediction|proba_0|proba_1|...
        """
        original_prediction_df = self._model_accessor.predict(self._original_test_df[:limit])
        new_prediciton_df = self._model_accessor.predict(new_df[:limit])
        proba_columns = [col for col in original_prediction_df.columns if 'proba_' in col]

        # move to % scale, it plays nicer with d3 ...
        original_prediction_df.loc[:, proba_columns] = np.around(original_prediction_df.loc[:, proba_columns] * 100)
        new_prediciton_df.loc[:, proba_columns] = np.around(new_prediciton_df.loc[:, proba_columns] * 100)

        return {FROM_ORIGINAL: original_prediction_df, FROM_NEW: new_prediciton_df}

    def _get_fugacity(self, prediction_dict):
        original_prediction_df = prediction_dict.get(FROM_ORIGINAL)
        new_prediciton_df = prediction_dict.get(FROM_NEW)

        original_fugacity = (100*original_prediction_df['prediction'].value_counts(normalize=True)).round(decimals=2).to_dict()
        new_fugacity = (100*new_prediciton_df['prediction'].value_counts(normalize=True)).round(decimals=2).to_dict()
        fugacity = []
        for key in original_fugacity.keys():
            temp_fugacity = {}
            new_key = "Predicted {} (%)".format(key)
            temp_fugacity[' Score'] = new_key
            temp_fugacity['Original test set'] = original_fugacity.pop(key)
            temp_fugacity['New test set'] = new_fugacity.pop(key)
            fugacity.append(temp_fugacity)
        return fugacity
