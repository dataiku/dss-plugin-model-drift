# -*- coding: utf-8 -*-
import sys
import os
import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from doctor_handler import get_saved_model_version_id, get_model_info_handler
from preprocessing import  Preprocessor
from model_metadata import ModelAccessor
logger = logging.getLogger(__name__)


class DriftAnalyzer:

    def __init__(self, model_accessor=None):
        self.model_accessor = model_accessor
        self.original_test_df = model_accessor.get_original_test_df()
        self.drift_target_column = 'dku_flag'
        self.test_X = None
        self.test_Y = None

    def check(self):
        if self.model_accessor is None:
            raise ValueError('ModelAccessor object is not specified.')

    def prepare_data_for_drift_model(self, new_test_df): 
        target = self.model_accessor.get_target()
        original_df = self.original_test_df.drop(target, axis=1)
        if target in new_test_df:
            new_df = new_test_df.drop(target, axis=1)
        else:
            new_df = new_test_df.copy()
            
        original_df[self.drift_target_column] = 'original'
        new_df[self.drift_target_column] = 'new'
        
        max_rows = min(original_df.shape[0],new_df.shape[0]) # Need a balanced sample
        # TODO should raise a warning/error if the number of rows to sample is way too low
        df = pd.concat([original_df.head(max_rows), new_df.head(max_rows)], sort=False)
        selected_features = [self.drift_target_column] + self.model_accessor.get_selected_features()
        return df.loc[:, selected_features]

    def train_drift_model(self, test_df):        
        df = self.prepare_data_for_drift_model(test_df)
        preprocessor = Preprocessor(df, target=self.drift_target_column)
        train, test = preprocessor.get_processed_train_test()
        
        train_X = train.drop(self.drift_target_column, axis=1)
        train_Y = np.array(train[self.drift_target_column])
        self.test_X = test.drop(self.drift_target_column, axis=1) # we will use them later when compute metrics
        self.test_Y = np.array(test[self.drift_target_column]) 
        
        drift_features = train_X.columns
        drift_clf = RandomForestClassifier(n_estimators=100, random_state=1337, max_depth=13, min_samples_leaf=1)
        drift_clf.fit(train_X, train_Y)
        return drift_features, drift_clf
        
    def _get_drift_feature_importance(self, drift_features, drift_clf):
        feature_importance = []
        for feature_name, feat_importance in zip(drift_features, drift_clf.feature_importances_):
            feature_importance.append({
                'feature': feature_name,
                'importance': feat_importance
            })
        return feature_importance

    def _get_feature_importance_metrics(self, drift_features, drift_clf):
        original_feature_importance = self.model_accessor.get_feature_importance()
        drift_feature_importance = self._get_drift_feature_importance(drift_features, drift_clf)
        return original_feature_importance, drift_feature_importance

    def _get_drift_auc(self, drift_clf):
        probas = drift_clf.predict_proba(self.test_X)
        test_Y_ser = pd.Series(self.test_Y)
        auc_score = roc_auc_score(test_Y_ser, probas[:, 1]) # only for binary classif
        return auc_score

    def _get_prediction(self, new_test_df):
        original_predictions = self.model_accessor.predict(self.original_test_df).values
        new_predicitons = self.model_accessor.predict(new_test_df).values
        return original_predictions, new_predicitons

    def generate_drift_metrics(self, new_test_df, drift_features, drift_clf):
        logger.info("Computing drift metrics ...")
        feature_importance_metrics = self._get_feature_importance_metrics(drift_features, drift_clf)
        drift_auc = self._get_drift_auc(drift_clf)
        prediction_metrics = self._get_prediction(new_test_df)
        return {'feature_importance': feature_importance_metrics, 'drift_auc': drift_auc, 'predictions': prediction_metrics}
