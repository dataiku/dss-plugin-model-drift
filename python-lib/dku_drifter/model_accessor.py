# -*- coding: utf-8 -*-
import pandas as pd
import logging
logger = logging.getLogger(__name__)

class ModelAccessor:
    def __init__(self, model_handler=None):
        self.model_handler = model_handler

    def check(self):
        if self.model_handler is None:
            raise ValueError('Model_handler object is not specified')

    def get_target_variable(self):
        return self.model_handler.get_target_variable()
        
    def get_original_test_df(self):
        return self.model_handler.get_test_df()[0]
    
    def get_per_feature(self):
        return self.model_handler.get_per_feature()
    
    def get_predictor(self):
        return self.model_handler.get_predictor()
    
    def get_feature_importance(self):
        predictor = self.get_predictor()
        clf = predictor._clf
        feature_importance = []
        x = clf.feature_importances_
        y = predictor.get_features
        for feature_name, feat_importance in zip(predictor.get_features(), clf.feature_importances_):
            feature_importance.append({
                'feature': feature_name,
                'importance': int(feat_importance)
            })
            
        dfx = pd.DataFrame(feature_importance).sort_values(by='importance', ascending=False).reset_index(drop=True).drop('importance', axis=1)
        return dfx.rename_axis('importance').reset_index().set_index('feature')
    
    def get_selected_features(self):
        selected_features = []
        for feat, feat_info in self.get_per_feature().items():
            if feat_info.get('role') == 'INPUT':
                selected_features.append(feat)
        return selected_features
    
    def predict(self, df):
        return self.get_predictor().predict(df)
