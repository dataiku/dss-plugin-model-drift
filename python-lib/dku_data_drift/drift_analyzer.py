# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from dku_data_drift.preprocessing import Preprocessor
from dku_data_drift.dataframe_helpers import not_enough_data
from dku_data_drift.model_tools import format_proba_density

logger = logging.getLogger(__name__)

ORIGIN_COLUMN = '__dku_row_origin__'  # name for the column that will contain the information from where the row is from (original test dataset or new dataframe)
FROM_ORIGINAL = 'original'
FROM_NEW = 'new'
MIN_NUM_ROWS = 500 # heuristic choice
MAX_NUM_ROW = 100000 # heuristic choice
CUMULATIVE_PERCENTAGE_THRESHOLD = 90
PREDICTION_TEST_SIZE = 10000


class DriftAnalyzer:

    def __init__(self, prediction_type=None, min_num_row=MIN_NUM_ROWS):
        self.prediction_type = prediction_type
        self.min_num_row = min_num_row
        self.drift_clf = RandomForestClassifier(n_estimators=100, random_state=1337, max_depth=13, min_samples_leaf=1)

        self._original_df = None
        self._new_df = None
        self._drift_test_X = None
        self._drift_test_Y = None
        self._model_accessor = None
        self.has_predictions = False
        self.target = None
        self.features_in_drift_model = None
        #self.check()

    def check(self):
        #if self._model_accessor.get_prediction_type() == 'CLUSTERING':
        #    raise ValueError('Clustering model is not supported.')
        return None

    def fit(self, new_df, model_accessor=None, original_df=None, target=None):
        """
        Trains a classifier that attempts to discriminate between rows from the provided dataframe and
        rows from the dataset originally used to evaluate the model

        Returns (columns, classifier)
        """
        logger.info("Preparing the drift model...")

        if target is not None and target not in new_df:
            raise ValueError('The new dataframe does not contain target "{}".'.format(target))

        if model_accessor is not None and original_df is not None:
            raise ValueError('model_accessor and original_df can not be defined at the same time. Please choose one of them.')

        if model_accessor is not None and original_df is None and target is None:
            self._model_accessor = model_accessor
            self.has_predictions = True
            self.target = self._model_accessor.get_target_variable()
            self.prediction_type = self._model_accessor.get_prediction_type()
            original_df = self._model_accessor.get_original_test_df()
            df = self.prepare_data_when_having_target(new_df, original_df)
        elif model_accessor is None and original_df is not None and target is not None:
            self.has_predictions = True
            self.target = target
            df = self.prepare_data_when_having_target(new_df, original_df)
        elif model_accessor is None and original_df is not None and target is None:
            df = self.prepare_data_when_without_target(new_df, original_df)
        else:
            raise NotImplementedError('You need to precise either a model accessor or an original df.')

        preprocessor = Preprocessor(df, target=ORIGIN_COLUMN)
        train, test = preprocessor.get_processed_train_test()
        drift_train_X = train.drop(ORIGIN_COLUMN, axis=1)
        drift_train_Y = np.array(train[ORIGIN_COLUMN])
        self._drift_test_X = test.drop(ORIGIN_COLUMN, axis=1)  # we will use them later when compute metrics
        self._drift_test_Y = np.array(test[ORIGIN_COLUMN])
        self.features_in_drift_model = drift_train_X.columns

        logger.info("Fitting the drift model...")
        self.drift_clf.fit(drift_train_X, drift_train_Y)

    def prepare_data_when_having_target(self, new_df, original_df):
        logger.info('Prepare data with target for drift model')
        self._new_df = new_df
        self._original_df = original_df
        new_df_without_target = new_df.drop(self.target, axis=1)
        original_df_without_target = original_df.drop(self.target, axis=1)
        return self._prepare_data_for_drift_model(new_df_without_target, original_df_without_target)


    def prepare_data_when_without_target(self, new_df, original_df):
        logger.info('Prepare data without target for drift model')
        return self._prepare_data_for_drift_model(new_df, original_df)

    def get_drift_metrics_for_webapp(self):
        """
        For the model view webapp, return format easy to plot viz

        :return:
        """

        if self.features_in_drift_model is None or self.drift_clf is None:
            logger.warning('drift_features and drift_clf must be defined')
            return {}

        logger.info("Computing drift metrics ...")
        drift_accuracy = self.get_drift_score()
        feature_importance_metrics = self._get_feature_importance_metrics()

        if self.prediction_type == 'REGRESSION':
            kde_dict = self._get_regression_prediction_metrics()
            fugacity_metrics = {}
            label_list = []
        elif self.prediction_type == 'CLASSIFICATION':
            kde_dict, fugacity_metrics, label_list = self._get_classification_prediction_metrics()
        else:
            raise ValueError('Prediction type not defined.')

        return {'type': self.prediction_type,
                'feature_importance': feature_importance_metrics,
                'drift_accuracy': drift_accuracy,
                'kde': kde_dict,
                'fugacity': fugacity_metrics,
                'label_list': label_list}

    def _get_classification_prediction_metrics(self):

        if not self.has_predictions:
            raise ValueError('DriftAnalyzer needs a target.')

        if self.prediction_type != 'CLASSIFICATION':
            raise ValueError('Can not use this function with a prediction type {}.'.format(self.prediction_type))

        if self._model_accessor is not None:
            prediction_dict = self.get_predictions_from_original_model(limit=PREDICTION_TEST_SIZE)
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
            fugacity_metrics = self.get_fugacity(reformat=True)
            label_list = [label for label in fugacity_metrics[0].keys() if label != 'source']

            return kde_dict, fugacity_metrics, label_list
        else:
            fugacity_metrics = self.get_fugacity()
            return None, fugacity_metrics, None

    def _get_regression_prediction_metrics(self):

        if not self.has_predictions:
            raise ValueError('No target was defined at fit phase.')

        if self.prediction_type != 'REGRESSION':
            raise ValueError('Can not use this function with a prediction type {}.'.format(self.prediction_type))

        prediction_dict = self.get_predictions_from_original_model(limit=PREDICTION_TEST_SIZE)
        kde_original = format_proba_density(prediction_dict.get(FROM_ORIGINAL).values)
        kde_new = format_proba_density(prediction_dict.get(FROM_NEW).values)
        kde_dict= {'Prediction': {FROM_ORIGINAL: kde_original, FROM_NEW: kde_new}} # to have the same format as in classif case
        return kde_dict


    def _prepare_data_for_drift_model(self, new_df, original_df):
        """
        Sampling function so that original test set and new test set has the same ratio in the drift training set
        For now only do top n sampling, with max n = MAX_NUM_ROW

        :return: a dataframe with data source target (orignal vs new)
        """

        if not_enough_data(new_df, min_len=self.min_num_row):
            raise ValueError('The new dataset is too small ({} rows) to have stable result, it needs to have at least {} rows'.format(len(new_df),self.min_num_row))

        if not_enough_data(original_df, min_len=self.min_num_row):
            raise ValueError('The original dataset is too small ({} rows) to have stable result, it needs to have at least {} rows'.format(len(original_df),self.min_num_row))

        original_df[ORIGIN_COLUMN] = FROM_ORIGINAL
        new_df[ORIGIN_COLUMN] = FROM_NEW

        logger.info("Rebalancing data:")
        number_of_rows = min(original_df.shape[0], new_df.shape[0], MAX_NUM_ROW)
        logger.info(" - original dataset had %s rows, new dataset has %s. Selecting the first %s for each." % (original_df.shape[0], new_df.shape[0], number_of_rows))

        df = pd.concat([original_df.head(number_of_rows), new_df.head(number_of_rows)], sort=False)

        if self._model_accessor is not None:
            selected_features = [ORIGIN_COLUMN] + self._model_accessor.get_selected_features()
        else:
            selected_features = original_df.columns

        missing_features = set(selected_features) - set(new_df.columns)
        if len(missing_features) > 0:
            raise ValueError('Missing column(s) in the new dataframe: {}'.format(', '.join(list(missing_features))))

        return df.loc[:, selected_features]

    def get_drift_feature_importance(self, cumulative_percentage_threshold=95):
        feature_importance = []
        for feature_name, feat_importance in zip(self.features_in_drift_model, self.drift_clf.feature_importances_):
            feature_importance.append({
                'feature': feature_name,
                'importance': 100 * feat_importance / sum(self.drift_clf.feature_importances_)
            })

        dfx = pd.DataFrame(feature_importance).sort_values(by='importance', ascending=False).reset_index(drop=True)
        dfx['cumulative_importance'] = dfx['importance'].cumsum()
        dfx_top = dfx.loc[dfx['cumulative_importance'] <= cumulative_percentage_threshold]
        return dfx_top.rename_axis('rank').reset_index().set_index('feature')

    def get_original_feature_importance(self, cumulative_percentage_threshold=95):
        if self._model_accessor is not None:
            return self._model_accessor.get_feature_importance(cumulative_percentage_threshold)
        else:
            raise ValueError('DriftAnalyzer needs a ModelAccessor as input.')

    def _get_feature_importance_metrics(self):
        """
        For visualisation purpose

        :return:
        """
        if self._model_accessor is None:
            drift_feature_importance_df = self.get_drift_feature_importance()
            topn_drift_feature = drift_feature_importance_df.to_dict()['importance']
            return {'drift_model': topn_drift_feature}
        else:
            original_feature_importance_df = self.get_original_feature_importance()
            drift_feature_importance_df = self.get_drift_feature_importance()
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
    
    def get_drift_score(self, output_raw_score=False):

        """
        Drift score is the accuracy of drift model

        :param output_raw_score:
        :return:
        """
        predicted_Y = self.drift_clf.predict(self._drift_test_X)
        test_Y = pd.Series(self._drift_test_Y)
        drift_accuracy = accuracy_score(test_Y, predicted_Y)
        if output_raw_score:
            return drift_accuracy
        else:
            exponential_function = lambda x: round(np.exp(1 - 1 / (np.power(x, 2.5))), 2)
            return exponential_function(drift_accuracy) # make the score looks more "logic" from the user point of view

    def get_predictions_from_original_model(self, limit=10000):
        """
        Predictions on the test set of original and new data

        The result of model_accessor.predict() is a dataframe prediction|proba_0|proba_1|...
        """
        if not self.has_predictions:
            raise ValueError('No target was defined at fit phase.')

        if self._model_accessor is not None:
            original_prediction_df = self._model_accessor.predict(self._original_df[:limit])
            original_prediction_df = original_prediction_df.rename(columns={'prediction':self.target})
            new_predicton_df = self._model_accessor.predict(self._new_df[:limit])
            new_predicton_df = new_predicton_df.rename(columns={'prediction':self.target})

            if self._model_accessor.get_prediction_type() == 'CLASSIFICATION':
                proba_columns = [col for col in original_prediction_df.columns if 'proba_' in col]
                # move to % scale, it plays nicer with d3 ...
                original_prediction_df.loc[:, proba_columns] = np.around(original_prediction_df.loc[:, proba_columns] * 100)
                new_predicton_df.loc[:, proba_columns] = np.around(new_predicton_df.loc[:, proba_columns] * 100)

            return {FROM_ORIGINAL: original_prediction_df, FROM_NEW: new_predicton_df}

        else: # no proba columns
            original_prediction_df = self._original_df.loc[:, [self.target]]
            new_prediciton_df = self._new_df.loc[:, [self.target]]
            return {FROM_ORIGINAL: original_prediction_df, FROM_NEW: new_prediciton_df}

    def get_fugacity(self, reformat=False):
        """
        For classification only, this compute the ratio of each predicted label

        :param prediction_dict:
        :return:
        """
        if self.prediction_type != 'CLASSIFICATION':
            raise ValueError('This function is for prediction of type CLASSIFICATION.'.format(self.prediction_type))

        if not self.has_predictions:
            raise ValueError('No target was defined in the fit phase.')

        prediction_dict = self.get_predictions_from_original_model()
        original_prediction_df = prediction_dict.get(FROM_ORIGINAL)
        new_prediciton_df = prediction_dict.get(FROM_NEW)

        if reformat: # for the model view
            original_fugacity = (100 * original_prediction_df[self.target].value_counts(normalize=True)).round(decimals=2).to_dict()
            new_fugacity = (100 * new_prediciton_df[self.target].value_counts(normalize=True)).round(decimals=2).to_dict()
            fugacity = []
            for key in original_fugacity.keys():
                temp_fugacity = {}
                new_key = "Predicted {} (%)".format(key)
                temp_fugacity[' Score'] = new_key
                temp_fugacity['Test dataset'] = original_fugacity.get(key, 0.)
                temp_fugacity['Input dataset'] = new_fugacity.get(key, 0.)
                fugacity.append(temp_fugacity)
            return fugacity
        else:
            original_fugacity = (100 * original_prediction_df[self.target].value_counts(normalize=True)).round(decimals=2).rename_axis('class').reset_index(name='percentage')
            new_fugacity = (100 * new_prediciton_df[self.target].value_counts(normalize=True)).round(decimals=2).rename_axis('class').reset_index(name='percentage')
            return original_fugacity, new_fugacity