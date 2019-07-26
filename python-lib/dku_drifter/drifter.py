# -*- coding: utf-8 -*-
import logging
import sys
import os
import pandas as pd
import numpy as np
import dataiku
from dataiku import pandasutils as pdu

from doctor_handler import get_saved_model_version_id, get_model_info_handler
from preprocessing import  Preprocessor

logger = logging.getLogger(__name__)

class Drifter:

    def __init__(self, model_id, new_test_set_name, new_target='dku_flag'):

        self.model_id = model_id
        self.new_test_set_name = new_test_set_name
        self.new_target = new_target
        self.new_test_df = dataiku.Dataset(self.new_test_set_name).get_dataframe()
        self.original_test_df = None

        self.model_handler = self._get_model_handler()
        self.drift_clf = None

        self.train_X = None
        self.train_Y = None
        self.test_X = None
        self.test_Y = None

    def _get_model_handler(self):
        my_data_dir = os.environ['DIP_HOME']
        saved_model_version_id = get_saved_model_version_id(self.model_id)
        model_handler = get_model_info_handler(saved_model_version_id, my_data_dir)
        return model_handler

    def concatenate_new_and_original_data(self):
        target = self.model_handler.get_target_variable()
        self.original_test_df = self.model_handler.get_test_df()[0].drop(target, axis=1)
        if target in self.new_test_df:
            self.new_test_df = self.new_test_df.drop(target, axis=1)

        self.original_test_df[self.new_target] = 'original'
        self.new_test_df[self.new_target] = 'new'
        df = pd.concat([self.original_test_df, self.new_test_df], sort=False)
        return df

    def _get_selected_features(self):
        selected_features = [self.new_target]
        for feat, feat_info in self.model_handler.get_per_feature().items():
            if feat_info.get('role') == 'INPUT':
                selected_features.append(feat)
        return selected_features

    def train_drift_model(self):

        df = self.concatenate_new_and_original_data()
        selected_features = self._get_selected_features()
        preprocessor = Preprocessor(df.loc[:, selected_features], target=self.new_target)
        train, test = preprocessor.get_processed_train_test()

        self.train_X = train.drop(self.new_target, axis=1)
        self.test_X = test.drop(self.new_target, axis=1)

        self.train_Y = np.array(train[self.new_target])
        self.test_Y = np.array(test[self.new_target])

        from sklearn.ensemble import RandomForestClassifier
        self.drift_clf = RandomForestClassifier(n_estimators=100, random_state=1337, max_depth=13, min_samples_leaf=1)
        self.drift_clf.fit(self.train_X, self.train_Y)

    def _get_feature_importance(self, clf, features):
        feature_importance = []
        for feature_name, feat_importance in zip(features, clf.feature_importances_):
            feature_importance.append({
                'feature': feature_name,
                'importance': feat_importance
            })

        return feature_importance

    def _get_feature_importance_metrics(self):

        original_predictor = self.model_handler.get_predictor()
        original_feature_importance = self._get_feature_importance(original_predictor._clf, original_predictor.get_features())
        drift_feature_importance = self._get_feature_importance(self.drift_clf, self.train_X.columns)

        return (original_feature_importance, drift_feature_importance)

    def _get_drift_auc(self):

        probas = self.drift_clf.predict_proba(self.test_X)
        from dataiku.doctor.utils.metrics import mroc_auc_score
        test_Y_ser = pd.Series(self.test_Y)
        auc_score = mroc_auc_score(test_Y_ser, probas)

        return auc_score

    def _get_prediction(self):

        original_precdictor = self.model_handler.get_predictor()
        original_predictions = original_precdictor.predict(self.original_test_df).values
        new_predicitons = original_precdictor.predict(self.new_test_df).values
        return (original_predictions, new_predicitons)

    def generate_drift_metrics(self):

        if self.drift_clf is None:
            logger.error('No drift model trained.')

        feature_importance_metrics = self._get_feature_importance_metrics()
        drift_auc = self._get_drift_auc()
        prediction_metrics = self._get_prediction()

        return {'feature_importance_metrics': feature_importance_metrics, 'drift_auc': drift_auc, 'prediction_metrics': prediction_metrics}


