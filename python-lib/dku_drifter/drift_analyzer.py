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
            
        dfx = pd.DataFrame(feature_importance).sort_values(by='importance', ascending=False).reset_index(drop=True).drop('importance', axis=1)
        return dfx.rename_axis('importance').reset_index().set_index('feature')
    
    def _get_feature_importance_metrics(self, drift_features, drift_clf, top_n):
        original_feature_importance_df = self.model_accessor.get_feature_importance()
        drift_feature_importance_df = self._get_drift_feature_importance(drift_features, drift_clf)
        
        topn_drift_feature = drift_feature_importance_df[:top_n].to_dict()['importance']
        topn_original_feature = original_feature_importance_df.loc[topn_drift_feature.keys()].to_dict()['importance']
        
        feature_importance_list = []
        for feature in topn_drift_feature.keys():
            feature_importance_info = {'original_model': topn_original_feature.get(feature), 'drift_model':topn_drift_feature.get(feature), 'feature': feature}
            feature_importance_list.append(feature_importance_info)
        
        return feature_importance_list 

    def _get_drift_auc(self, drift_clf):
        probas = drift_clf.predict_proba(self.test_X)
        test_Y_ser = pd.Series(self.test_Y)
        auc_score = mroc_auc_score(test_Y_ser, probas) 
        return auc_score
    
    def _get_drift_accuracy(self, drift_clf):
        predicted_Y = drift_clf.predict(self.test_X)
        test_Y = pd.Series(self.test_Y)
        drift_accuracy = round(accuracy_score(test_Y, predicted_Y),2)
        return drift_accuracy
    
    def _get_predictions(self, new_df, limit=500):     
        """
        The result of model_accessor.predict() is a dataframe prediction|proba_0|proba_1|...
        """
                
        original_prediction_df = self.model_accessor.predict(self.original_df[:limit])
        new_prediciton_df = self.model_accessor.predict(new_df[:limit])
        proba_columns = [col for col in original_prediction_df.columns if 'proba_' in col] 

        # move to % scale, it plays nicer with d3 ...
        original_prediction_df.loc[:, proba_columns] = original_prediction_df.loc[:, proba_columns] * 100
        new_prediciton_df.loc[:, proba_columns] = new_prediciton_df.loc[:, proba_columns] * 100 

        return {'original': original_prediction_df, 'new': new_prediciton_df}
    
    def _get_frugacity(self, prediction_dict):
        
        original_prediction_df = prediction_dict.get('original')
        new_prediciton_df = prediction_dict.get('new')
        
        original_fugacity = (100*original_prediction_df['prediction'].value_counts(normalize=True)).round(decimals=2).to_dict()
        for key in original_fugacity.keys():
            new_key = "Label {} (in %)".format(key)
            original_fugacity[new_key] = original_fugacity.pop(key)
        original_fugacity['source'] = 'Original test set'
        
        new_fugacity = (100*new_prediciton_df['prediction'].value_counts(normalize=True)).round(decimals=2).to_dict()
        for key in new_fugacity.keys():
            new_key = "Label {} (in %)".format(key)
            new_fugacity[new_key] = new_fugacity.pop(key)
        new_fugacity['source'] = 'New test set'
        
        return [original_fugacity, new_fugacity]

    
    def _get_stat_test2(self, prediction_dict, alpha=0.05):
        
        original_prediction_df = prediction_dict.get('original')
        new_prediction_df = prediction_dict.get('new')
        proba_columns = [col for col in original_prediction_df.columns if 'proba_' in col] 
        
        stat_test_dict = {}
        
        for column in proba_columns:
            original_probas = original_prediction_df[column].values
            new_probas = new_prediction_df[column].values
            t_test = stats.ttest_ind(original_probas, new_probas, equal_var=False)[-1]
            stat_test_dict[column] = round(t_test, 2)
            
        return stat_test_dict

    def _get_stat_test(self, x,y, alpha=0.05):
        '''return p-values for t-test, ks-test and anderson-test'''
        t_test = stats.ttest_ind(x,y, equal_var=False)[-1]
        #ks_test = stats.ks_2samp(x,y)[-1]
        #and_test = stats.anderson_ksamp([x,y])[-1]
        pvals = {"t_test": round(t_test,4)}#,"ks_test":ks_test,"and_test":and_test} 
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
        feature_importance_metrics = self._get_feature_importance_metrics(drift_features, drift_clf, top_n=5)
        drift_auc = self._get_drift_auc(drift_clf)
        drift_accuracy = self._get_drift_accuracy(drift_clf)
        prediction_dict = self._get_predictions(new_df, limit=500)
        # for now we take only class 1
        class_1_original = np.around(prediction_dict.get('original')['proba_1'].values,2).tolist()
        class_1_new = np.around(prediction_dict.get('new')['proba_1'].values,2).tolist()
        predictions = {'original': class_1_original, 'new': class_1_new}
        
        fugacity_metrics = self._get_frugacity(prediction_dict)
        #stat_metrics = self._get_stat_test(prediction_metrics.get('original'), prediction_metrics.get('new'))
        stat_metrics = self._get_stat_test2(prediction_dict)
        print('doneeeeee')
        return {'feature_importance': feature_importance_metrics, 'drift_accuracy': drift_accuracy,'predictions': predictions, 'stat_metrics':stat_metrics, 'fugacity': fugacity_metrics}
