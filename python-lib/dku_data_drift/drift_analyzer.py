#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Drift Analyzer main class definition.
"""

import joblib
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

from dku_data_drift.preprocessing import Preprocessor
from dku_data_drift.dataframe_helpers import not_enough_data
from dku_data_drift.model_tools import format_proba_density

# TODO: Remove after usage
# Path for object storage
from dev_helper import OBJECT_PATH, DATA_PATH

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Model Drift Plugin | %(levelname)s - %(message)s')

# Name of the new column for the origin
ORIGIN_COLUMN = '__dku_row_origin__'
FROM_ORIGINAL = 'original'
FROM_NEW = 'new'
MIN_NUM_ROWS = 1000
ALGORITHMS_WITH_VARIABLE_IMPORTANCE = [RandomForestClassifier, GradientBoostingClassifier,
                                       ExtraTreesClassifier, DecisionTreeClassifier]
CUMULATIVE_PERCENTAGE_THRESHOLD = 90
PREDICTION_TEST_SIZE = 10000


class DriftAnalyzer:
    """
    Analyse the drift between two datasets. It can contain a model associated to the original dataset.
    If a model is defined, we use its predictions to analyse drift.
    """
    def __init__(self, model_accessor):
        """
        :param: model_accessor
        :example:
        """
        self._model_accessor = model_accessor
        joblib.dump(self._model_accessor, OBJECT_PATH+'_model_accessor')
        self._original_test_df = model_accessor.get_original_test_df()
        joblib.dump(self._original_test_df, OBJECT_PATH+'_original_test_df')
        self._new_test_df = None
        self._test_X = None
        self._test_Y = None
        self.check()

    def check(self):
        """
        Check if the model supports drift analysis task
        """
        if self._model_accessor.get_prediction_type() == 'CLUSTERING':
            raise ValueError('Clustering model is not supported.')

    def train_drift_model(self, new_test_df, min_num_row=MIN_NUM_ROWS):
        """
        Create a dataframe associated with the new data.
        Apply preprocessing to retrieve train and test set.
        :param: new_test_df
        :param: min_num_row
        :return:
            train_X.columns: type <list>
            :example: ['column1', 'column2', ...]
            clf: type Scikit Learn Model

        Returns (columns, classifier)
        """
        logger.info("Preparing the drift model...")
        joblib.dump(new_test_df, OBJECT_PATH+'new_test_df')
        df = self._prepare_data_for_drift_model(new_test_df)
        preprocessor = Preprocessor(df, target=ORIGIN_COLUMN)
        train, test = preprocessor.get_processed_train_test()

        if not_enough_data(df, min_len=min_num_row):
            raise ValueError(
                'Either the original test dataset or the new input dataset is too small, they each need to have at least {} rows'.format(
                    min_num_row / 2))

        # Combine the two datasets rows that are in the dataset
        train_X = train.drop(ORIGIN_COLUMN, axis=1)
        train_Y = np.array(train[ORIGIN_COLUMN])
        self._test_X = test.drop(ORIGIN_COLUMN, axis=1)  # we will use them later when compute metrics
        self._test_Y = np.array(test[ORIGIN_COLUMN])
        self._new_test_df = new_test_df

        # Fit a random classifier to the data
        clf = RandomForestClassifier(n_estimators=100, random_state=1337, max_depth=13, min_samples_leaf=1)
        logger.info("Fitting the drift model...")
        clf.fit(train_X, train_Y)
        joblib.dump(clf, OBJECT_PATH+'clf')
        return train_X.columns, clf

    def compute_drift_metrics(self, drift_features, drift_clf):
        """
        Compute the drift metrics

        :param: drift_features
        :param: drift_clf

        :return: drift_metrics : Type <dict>
        Contains all the retrieved metrics for model performance assessment
        """
        joblib.dump(drift_features, OBJECT_PATH+"drift_features")
        joblib.dump(drift_clf, OBJECT_PATH+"drift_clf")
        if drift_features is None or drift_clf is None:
            logger.warning('drift_features and drift_clf must be defined')
            return {}

        logger.info("Computing drift metrics ...")
        drift_accuracy = self._get_drift_accuracy(drift_clf)
        feature_importance_metrics = self._get_feature_importance_metrics(drift_features, drift_clf)

        if self._model_accessor.get_prediction_type() == 'REGRESSION':
            kde_dict = self._get_regression_prediction_metrics()
            fugacity_metrics = {}
            label_list = []
        else:
            kde_dict, fugacity_metrics, label_list = self._get_classification_prediction_metrics()

        drift_metrics =  {'type': self._model_accessor.get_prediction_type(),
                'feature_importance': feature_importance_metrics,
                'drift_accuracy': drift_accuracy,
                'kde': kde_dict,
                'fugacity': fugacity_metrics,
                'label_list': label_list}

        return drift_metrics

    def _get_classification_prediction_metrics(self):
        """
        Collect prediction metrics for a classification task
        """
        prediction_dict = self._get_predictions(limit=PREDICTION_TEST_SIZE)
        predictions_by_class = {}
        for label in prediction_dict.get(FROM_ORIGINAL).columns:
            if 'proba_' in label:
                original_proba = np.around(prediction_dict.get(FROM_ORIGINAL)[label].values, 2).tolist()
                new_proba = np.around(prediction_dict.get(FROM_NEW)[label].values, 2).tolist()
                predictions_by_class[label] = {FROM_ORIGINAL: original_proba, FROM_NEW: new_proba}
        kde_dict = {}
        for label in predictions_by_class.keys():
            kde_original = format_proba_density(predictions_by_class.get(label).get(FROM_ORIGINAL))
            kde_new = format_proba_density(predictions_by_class.get(label).get(FROM_NEW))
            cleaned_label = label.replace('proba_', 'Class ')
            kde_dict[cleaned_label] = {FROM_ORIGINAL: kde_original, FROM_NEW: kde_new}
        fugacity_metrics = self._get_fugacity(prediction_dict)
        label_list = [label for label in fugacity_metrics[0].keys() if label != 'source']

        return kde_dict, fugacity_metrics, label_list

    def _get_regression_prediction_metrics(self):
        """
        Collect performance metrics for regression or prediction tasks
        :return: kde_dict
        """
        prediction_dict = self._get_predictions(limit=PREDICTION_TEST_SIZE)
        kde_original = format_proba_density(prediction_dict.get(FROM_ORIGINAL).values)
        kde_new = format_proba_density(prediction_dict.get(FROM_NEW).values)

        kde_dict = {'Prediction': {FROM_ORIGINAL: kde_original,
                                   FROM_NEW: kde_new}}  # to have the same format as in classif case

        return kde_dict

    def _prepare_data_for_drift_model(self, new_test_df):
        """
        Sampling function so that original test set and new test set has the same ratio in the drift training set
        For now only do top n sampling

        :param new_test_df:
        :return:
        """
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
        logger.info(" - original test dataset had %s rows, new dataframe has %s. Selecting %s for each." % (
        original_df.shape[0], new_df.shape[0], number_of_rows))

        df = pd.concat([original_df.head(number_of_rows), new_df.head(number_of_rows)], sort=False)
        selected_features = [ORIGIN_COLUMN] + self._model_accessor.get_selected_features()
        missing_features = set(selected_features) - set(new_df.columns)
        if len(missing_features) > 0:
            raise ValueError('Missing columns in the new test set: {}'.format(', '.join(list(missing_features))))

        return df.loc[:, selected_features]

    def _get_drift_feature_importance(self, drift_features, drift_clf, cumulative_percentage_threshold=95):
        """
        Get the feature importance for the drift model.
        :param: drift_features
        :param: drift_clf
        :param: cumulative_percentage_threshold
        """
        feature_importance = []
        for feature_name, feat_importance in zip(drift_features, drift_clf.feature_importances_):
            feature_importance.append({
                'feature': feature_name,
                'importance': 100 * feat_importance / sum(drift_clf.feature_importances_)
            })

        dfx = pd.DataFrame(feature_importance).sort_values(by='importance', ascending=False).reset_index(drop=True)
        dfx['cumulative_importance'] = dfx['importance'].cumsum()
        dfx_top = dfx.loc[dfx['cumulative_importance'] <= cumulative_percentage_threshold]
        return dfx_top.rename_axis('rank').reset_index().set_index('feature')

    def _get_feature_importance_metrics(self, drift_features, drift_clf):
        """
        Get the metrics for the largest feature importance
        :param: drift_features
        :param: drift_clf
        :return: feature_importance_metrics
        """
        original_feature_importance_df = self._model_accessor.get_feature_importance()
        drift_feature_importance_df = self._get_drift_feature_importance(drift_features, drift_clf,
                                                                         cumulative_percentage_threshold=95)
        topn_drift_feature = drift_feature_importance_df.to_dict()['importance']
        topn_original_feature = original_feature_importance_df.to_dict()['importance']
        feature_importance_metrics = []
        for feature in set(topn_original_feature.keys()).union(set(topn_drift_feature.keys())):
            drift_feat_rank = topn_drift_feature.get(feature)
            original_feat_rank = topn_original_feature.get(feature)
            if drift_feat_rank is None:
                logger.warn(
                    'Feature {} does not exist in the most important features of the drift model.'.format(feature))
            if original_feat_rank is None:
                logger.warn(
                    'Feature {} does not exist in the most important features of the orignal model.'.format(feature))
            feature_importance_metrics.append({
                'original_model': original_feat_rank if original_feat_rank else 0.01,
                'drift_model': drift_feat_rank if drift_feat_rank else 0.01,
                'feature': feature
            })
        return feature_importance_metrics

    def _exponential_function(self, score):
        """
        Get a round metric for the drift score.
        """
        return round(np.exp(1 - 1 / (np.power(score, 2.5))), 2)

    def _get_drift_accuracy(self, drift_clf):
        """
        Score the drift accuracy
        """
        predicted_Y = drift_clf.predict(self._test_X)
        test_Y = pd.Series(self._test_Y)
        drift_accuracy = accuracy_score(test_Y, predicted_Y)
        return self._exponential_function(drift_accuracy)

    def _get_predictions(self, limit=10000):
        """
        The result of model_accessor.predict() is a dataframe prediction|proba_0|proba_1|...
        """
        original_prediction_df = self._model_accessor.predict(self._original_test_df[:limit])
        new_prediciton_df = self._model_accessor.predict(self._new_test_df[:limit])

        if self._model_accessor.get_prediction_type() == 'CLASSIFICATION':
            proba_columns = [col for col in original_prediction_df.columns if 'proba_' in col]

            # move to % scale, it plays nicer with d3 ...
            original_prediction_df.loc[:, proba_columns] = np.around(original_prediction_df.loc[:, proba_columns] * 100)
            new_prediciton_df.loc[:, proba_columns] = np.around(new_prediciton_df.loc[:, proba_columns] * 100)

        return {FROM_ORIGINAL: original_prediction_df, FROM_NEW: new_prediciton_df}

    def _get_fugacity(self, prediction_dict):
        """
        For classification only, this compute the ratio of each predicted label

        :param prediction_dict:
        :return:
        """
        original_prediction_df = prediction_dict.get(FROM_ORIGINAL)
        new_prediciton_df = prediction_dict.get(FROM_NEW)

        original_fugacity = (100 * original_prediction_df['prediction'].value_counts(normalize=True)).round(
            decimals=2).to_dict()
        new_fugacity = (100 * new_prediciton_df['prediction'].value_counts(normalize=True)).round(decimals=2).to_dict()
        fugacity = []
        for key in original_fugacity.keys():
            temp_fugacity = {}
            new_key = "Predicted {} (%)".format(key)
            temp_fugacity[' Score'] = new_key
            temp_fugacity['Test dataset'] = original_fugacity.get(key, 0.)
            temp_fugacity['Input dataset'] = new_fugacity.get(key, 0.)
            fugacity.append(temp_fugacity)
        return fugacity

    def _algorithm_is_supported(self, predictor):
        """
        Check that the model enables the drift analysis
        """
        algo = predictor._clf
        for algorithm in ALGORITHMS_WITH_VARIABLE_IMPORTANCE:
            if isinstance(algo, algorithm):
                return True
            elif predictor.params.modeling_params.get('algorithm') == 'XGBOOST_CLASSIFICATION':
                return True
        return False
