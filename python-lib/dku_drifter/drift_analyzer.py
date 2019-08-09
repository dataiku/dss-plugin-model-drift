# -*- coding: utf-8 -*-
import sys
import os
import pandas as pd
import numpy as np
import logging
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from doctor_handler import get_saved_model_version_id, get_model_info_handler
from preprocessing import  Preprocessor
from model_accessor import ModelAccessor
from model_tools import mroc_auc_score
logger = logging.getLogger(__name__)


class DriftAnalyzer:

    def __init__(self, model_accessor, drift_target_column='dku_flag'):
        self.model_accessor = model_accessor
        self.original_df = model_accessor.get_original_test_df()
        self.drift_target_column = drift_target_column
        self.test_X = None
        self.test_Y = None

    def check(self):
        #if self.model_accessor is None:
        #    raise ValueError('ModelAccessor object is not specified.')
        pass

    def prepare_data_for_drift_model(self, new_test_df): 
        target = self.model_accessor.get_target_variable()
        original_df = self.original_df.drop(target, axis=1)
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

    def train_drift_model(self, new_df):
        df = self.prepare_data_for_drift_model(new_df)
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
        return pd.DataFrame(feature_importance).set_index('feature').sort_values(by='importance', ascending=False)

    def _get_feature_importance_metrics(self, drift_features, drift_clf, top_n):
        original_feature_importance_df = self.model_accessor.get_feature_importance()
        drift_feature_importance_df = self._get_drift_feature_importance(drift_features, drift_clf)
        
        topn_drift_feature = drift_feature_importance_df[:top_n]
        topn_original_feature = original_feature_importance_df.loc[topn_drift_feature.index.tolist()]
        
        return {'original_model': topn_original_feature.to_dict()['importance'], 'drift_model': topn_drift_feature.to_dict()['importance']}

    def _get_drift_auc(self, drift_clf):
        probas = drift_clf.predict_proba(self.test_X)
        test_Y_ser = pd.Series(self.test_Y)
        auc_score = mroc_auc_score(test_Y_ser, probas) 
        return auc_score
    
    def _get_drift_accuracy(self, drift_clf):
        predicted_Y = drift_clf.predict(self.test_X)
        test_Y = pd.Series(self.test_Y)
        drift_accuracy = accuracy_score(test_Y, predicted_Y) 
        return drift_accuracy

    def _get_predictions(self, new_test_df):
        # Take only the proba of class 1
        original_predictions = [x[-1] for x in self.model_accessor.predict(self.original_df).values.tolist()]
        new_predicitons = [x[-1] for x in self.model_accessor.predict(new_test_df).values.tolist()]
        return original_predictions, new_predicitons
    
    def _get_stat_test(self, x,y, alpha=0.05):
        '''return p-values for t-test, ks-test and anderson-test'''
        t_test = stats.ttest_ind(x,y, equal_var=False)[-1]
        ks_test = stats.ks_2samp(x,y)[-1]
        and_test = stats.anderson_ksamp([x,y])[-1]
        pvals = {"t_test":t_test,"ks_test":ks_test,"and_test":and_test} 
        h0= True
        for test,pval in pvals.items():
            if pval < alpha:
                logger.info("Independence hypothesis is rejected by %s at %.2f level"%(test,alpha))
                h0 = False
        if h0:
            logger.info("According to the data, the independence hypothesis is validated by all tests.")
        return pvals
    
    def generate_drift_metrics(self, new_df, drift_features, drift_clf):
        logger.info("Computing drift metrics ...")
        feature_importance_metrics = self._get_feature_importance_metrics(drift_features, drift_clf, 10)
        drift_auc = self._get_drift_auc(drift_clf)
        drift_accuracy = self._get_drift_accuracy(drift_clf)
        prediction_metrics = self._get_predictions(new_df)
        stat_metrics = self._get_stat_test(prediction_metrics[0], prediction_metrics[1])
        return {'feature_importance': feature_importance_metrics, 'drift_auc': drift_auc, 'drift_accuracy': drift_accuracy,'predictions': prediction_metrics, 'stat_metrics':stat_metrics}
