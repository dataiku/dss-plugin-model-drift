# -*- coding: utf-8 -*-
import logging
import math
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

from dku_data_drift.preprocessing import Preprocessor
from dku_data_drift.dataframe_helpers import not_enough_data

logger = logging.getLogger(__name__)

ORIGIN_COLUMN = '__dku_row_origin__'  # name for the column that will contain the information from where the row is from (original test dataset or new dataframe)
FROM_ORIGINAL = 'original'
FROM_NEW = 'new'
MIN_NUM_ROWS = 1000 # heuristic choice
ALGORITHMS_WITH_VARIABLE_IMPORTANCE = [RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, DecisionTreeClassifier]
CUMULATIVE_PERCENTAGE_THRESHOLD = 90
PREDICTION_TEST_SIZE = 10000

class DriftAnalyzer:

    def __init__(self, model_accessor):
        self._model_accessor = model_accessor
        self._original_test_df = model_accessor.get_original_test_df()
        self._new_test_df = None
        self._test_X = None
        self._test_Y = None
        self.check()

    def check(self):
        predictor = self._model_accessor.get_predictor()
        if not self._algorithm_is_supported(predictor):
            raise ValueError('{} is not a supported algorithm. Please choose one that has feature importances (tree-based models).'.format(predictor._clf.__module__))

    def train_drift_model(self, new_test_df, min_num_row=MIN_NUM_ROWS):
        """
        Trains a classifier that attempts to discriminate between rows from the provided dataframe and
        rows from the dataset originally used to evaluate the model

        Returns (columns, classifier)
        """
        logger.info("Preparing the drift model...")

        df = self._prepare_data_for_drift_model(new_test_df)
        preprocessor = Preprocessor(df, target=ORIGIN_COLUMN)
        train, test = preprocessor.get_processed_train_test()
        
        if not_enough_data(df, min_len=min_num_row):
            raise ValueError('The processed dataset has less than {} rows, not enough to train drift model.'.format(min_num_row))

        train_X = train.drop(ORIGIN_COLUMN, axis=1)
        train_Y = np.array(train[ORIGIN_COLUMN])
        self._test_X = test.drop(ORIGIN_COLUMN, axis=1)  # we will use them later when compute metrics
        self._test_Y = np.array(test[ORIGIN_COLUMN])
        self._new_test_df = new_test_df

        clf = RandomForestClassifier(n_estimators=100, random_state=1337, max_depth=13, min_samples_leaf=1)
        logger.info("Fitting the drift model...")
        clf.fit(train_X, train_Y)
        return train_X.columns, clf

    def compute_drift_metrics(self, drift_features, drift_clf):

        if drift_features is None or drift_clf is None:
            logger.warning('drift_features and drift_clf must be defined')
            return {}

        logger.info("Computing drift metrics ...")
        feature_importance_metrics = self._get_feature_importance_metrics(drift_features, drift_clf)
        drift_accuracy = self._get_drift_accuracy(drift_clf)
        prediction_dict = self._get_predictions(limit=PREDICTION_TEST_SIZE)

        predictions_by_class = {}
        for label in prediction_dict.get(FROM_ORIGINAL).columns:
            if 'proba_' in label:
                original_proba = np.around(prediction_dict.get(FROM_ORIGINAL)[label].values, 2).tolist()
                new_proba = np.around(prediction_dict.get(FROM_NEW)[label].values, 2).tolist()
                predictions_by_class[label] = {FROM_ORIGINAL: original_proba, FROM_NEW: new_proba}
        kde_dict = self._get_kde(predictions_by_class)
        fugacity_metrics = self._get_fugacity(prediction_dict)
        label_list = [label for label in fugacity_metrics[0].keys() if label != 'source']
        return {'feature_importance': feature_importance_metrics, 'drift_accuracy': drift_accuracy, 'kde': kde_dict,
                'fugacity': fugacity_metrics, 'label_list': label_list}

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
        missing_features = set(selected_features) - set(new_df.columns)
        if len(missing_features) > 0:
            raise ValueError('Missing columns in the new test set: {}'.format(', '.join(list(missing_features))))

        return df.loc[:, selected_features]

    def _get_kde(self, predictions):
        kde_dict = {}
        for label in predictions.keys():
            kde_original = self._format_proba_density(predictions.get(label).get(FROM_ORIGINAL))
            kde_new = self._format_proba_density(predictions.get(label).get(FROM_NEW))
            cleaned_label = label.replace('proba_', 'Class ')
            kde_dict[cleaned_label] = {FROM_ORIGINAL: kde_original, FROM_NEW: kde_new}
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
        X_plot = np.linspace(0, 100, 500, dtype=int)[:, np.newaxis]
        kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(data.reshape(-1, 1), sample_weight=sample_weight)
        Y_plot = np.exp(kde.score_samples(X_plot))
        Y_plot = [v if not np.isnan(v) else 0 for v in np.exp(kde.score_samples(X_plot))]
        return list(zip(X_plot.ravel(), Y_plot))

    def _get_drift_feature_importance(self, drift_features, drift_clf, cumulative_percentage_threshold=95):
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
        original_feature_importance_df = self._model_accessor.get_feature_importance()
        drift_feature_importance_df = self._get_drift_feature_importance(drift_features, drift_clf, cumulative_percentage_threshold=95)
        topn_drift_feature = drift_feature_importance_df.to_dict()['importance']
        topn_original_feature = original_feature_importance_df.to_dict()['importance']
        feature_importance_metrics = []
        for feature in set(topn_original_feature.keys()).union(set(topn_drift_feature.keys())):
            drift_feat_rank = topn_drift_feature.get(feature)
            original_feat_rank = topn_original_feature.get(feature)
            if drift_feat_rank is None:
                logger.warn('Feature {} does not exist in the most important features of the drift model.'.format(feature))
            if original_feat_rank is None:
                logger.warn('Feature {} does not exist in the most important features of the orignal model.'.format(feature))            
            feature_importance_metrics.append({
                'original_model': original_feat_rank if original_feat_rank else 0.01, 
                 'drift_model': drift_feat_rank if drift_feat_rank else 0.01,
                 'feature': feature
            })
        return feature_importance_metrics

    def _exponential_function(self, score): 
        return round(np.exp(1 - 1/(np.power(score, 2.5))),2)
    
    def _get_drift_accuracy(self, drift_clf):
        predicted_Y = drift_clf.predict(self._test_X)
        test_Y = pd.Series(self._test_Y)
        drift_accuracy = accuracy_score(test_Y, predicted_Y)
        return self._exponential_function(drift_accuracy) # make the score looks more "logic" from the user point of view

    def _get_predictions(self, limit=10000):
        """
        The result of model_accessor.predict() is a dataframe prediction|proba_0|proba_1|...
        """
        original_prediction_df = self._model_accessor.predict(self._original_test_df[:limit])
        new_prediciton_df = self._model_accessor.predict(self._new_test_df[:limit])
        proba_columns = [col for col in original_prediction_df.columns if 'proba_' in col]

        # move to % scale, it plays nicer with d3 ...
        original_prediction_df.loc[:, proba_columns] = np.around(original_prediction_df.loc[:, proba_columns] * 100)
        new_prediciton_df.loc[:, proba_columns] = np.around(new_prediciton_df.loc[:, proba_columns] * 100)

        return {FROM_ORIGINAL: original_prediction_df, FROM_NEW: new_prediciton_df}

    def _get_fugacity(self, prediction_dict):
        original_prediction_df = prediction_dict.get(FROM_ORIGINAL)
        new_prediciton_df = prediction_dict.get(FROM_NEW)

        original_fugacity = (100 * original_prediction_df['prediction'].value_counts(normalize=True)).round(decimals=2).to_dict()
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
        algo = predictor._clf
        for algorithm in ALGORITHMS_WITH_VARIABLE_IMPORTANCE:
            if isinstance(algo, algorithm):
                return True
            elif predictor.params.modeling_params.get('algorithm') == 'XGBOOST_CLASSIFICATION':
                return True
        return False