# -*- coding: utf-8 -*-
import sys
import os
import pandas as pd
import json
import numpy as np
import logging
import math
from scipy import stats
from statsmodels.stats.power import TTestIndPower
from sklearn.neighbors import KernelDensity
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
                'importance': 100*feat_importance/sum(drift_clf.feature_importances_)
            })
            
        dfx = pd.DataFrame(feature_importance).sort_values(by='importance', ascending=False).reset_index(drop=True)#.drop('importance', axis=1)
        return dfx.rename_axis('rank').reset_index().set_index('feature')
    
    def _get_feature_importance_metrics(self, drift_features, drift_clf, top_n):
        original_feature_importance_df = self.model_accessor.get_feature_importance()
        drift_feature_importance_df = self._get_drift_feature_importance(drift_features, drift_clf)
        
        topn_drift_feature = drift_feature_importance_df[:top_n].to_dict()['importance']
        topn_original_feature = original_feature_importance_df.loc[topn_drift_feature.keys()].to_dict()['importance']
        
        feature_importance_list = []
        for feature in topn_original_feature.keys():
            drift_feat_rank = topn_drift_feature.get(feature)
            if not np.isnan(drift_feat_rank):
                feature_importance_info = {'original_model':topn_original_feature.get(feature, 0), 'drift_model':drift_feat_rank, 'feature':feature}
                feature_importance_list.append(feature_importance_info)
            else:
                logger.warn('Feature {} does not exist in the orignal test set.'.format(feature))
        
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
    
    def _get_predictions(self, new_df, limit=10000):     
        """
        The result of model_accessor.predict() is a dataframe prediction|proba_0|proba_1|...
        """
                
        original_prediction_df = self.model_accessor.predict(self.original_df[:limit])
        new_prediciton_df = self.model_accessor.predict(new_df[:limit])
        proba_columns = [col for col in original_prediction_df.columns if 'proba_' in col] 

        # move to % scale, it plays nicer with d3 ...
        original_prediction_df.loc[:, proba_columns] = np.around(original_prediction_df.loc[:, proba_columns] * 100)
        new_prediciton_df.loc[:, proba_columns] = np.around(new_prediciton_df.loc[:, proba_columns] * 100)

        return {'original': original_prediction_df, 'new': new_prediciton_df}
    
    def _get_frugacity(self, prediction_dict):
        
        original_prediction_df = prediction_dict.get('original')
        new_prediciton_df = prediction_dict.get('new')
        
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

    def _get_stat_test3(self, kde_dict, alpha=0.05):
        
        power_analysis = TTestIndPower()
        stat_test_dict = {}
        for label in kde_dict.keys():
            kde_original = [x[1] for x in kde_dict.get(label).get('original')]
            kde_new = [x [1] for x in kde_dict.get(label).get('new')]
            # this effect size has an equal variance assumption (?)
            effect_size = np.abs((np.mean(kde_original) - np.mean(kde_new)))/np.std(kde_original)
            power = power_analysis.power(effect_size=effect_size, nobs1=len(kde_original), alpha=0.05)
            t_test = stats.ttest_ind(kde_original, kde_new, equal_var=False)[-1]
            stat_test_dict[label] = {'t_test': round(t_test,4), 'power': round(power,4)}
        
        return stat_test_dict

    
    def format_proba_density(self, data, sample_weight=None):
        """
        https://github.com/dataiku/dip/blob/e14c5b4f853081a3d481e9c13e71ea524ec9eec0/src/main/python/dataiku/doctor/prediction/classification_scoring.py#L392
        """
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
    
    def get_kde(self, predictions):
        kde_dict = {}
        for label in predictions.keys():
            kde_original = self.format_proba_density(predictions.get(label).get('original'))  
            kde_new = self.format_proba_density(predictions.get(label).get('new')) 
            kde_dict[label] = {'original':kde_original, 'new':kde_new}
        return kde_dict
    
    def generate_drift_metrics(self, new_df, drift_features, drift_clf):
        logger.info("Computing drift metrics ...")
        feature_importance_metrics = self._get_feature_importance_metrics(drift_features, drift_clf, top_n=50)
        drift_auc = self._get_drift_auc(drift_clf) 
        drift_accuracy = self._get_drift_accuracy(drift_clf)
        prediction_dict = self._get_predictions(new_df, limit=10000)
        
        predictions_by_class = {}
        for label in prediction_dict.get('original').columns:
            if 'proba_' in label:
                original_proba = np.around(prediction_dict.get('original')[label].values,2).tolist()
                new_proba = np.around(prediction_dict.get('new')[label].values,2).tolist()
                predictions_by_class[label] = {'original': original_proba, 'new': new_proba}
        
        kde_dict = self.get_kde(predictions_by_class)
        fugacity_metrics = self._get_frugacity(prediction_dict)
        stat_metrics = self._get_stat_test3(kde_dict)
        label_list = [label for label in fugacity_metrics[0].keys() if label != 'source']
        return {'feature_importance': feature_importance_metrics, 'drift_accuracy': drift_accuracy,'kde': kde_dict, 'stat_metrics':stat_metrics, 'fugacity': fugacity_metrics, 'label_list': label_list}
