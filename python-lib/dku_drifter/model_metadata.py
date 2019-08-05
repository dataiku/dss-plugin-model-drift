# -*- coding: utf-8 -*-
import logging
logger = logging.getLogger(__name__)

class ModelAccessor:
    def __init__(self, model_handler):
        self.model_handler = model_handler
        
    def get_target(self):
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
        for feature_name, feat_importance in zip(predictor.get_features(), clf.feature_importances_):
            feature_importance.append({
                'feature': feature_name,
                'importance': feat_importance
            })
        return feature_importance
    
    def get_selected_features(self):
        selected_features = []
        for feat, feat_info in self.get_per_feature().items():
            if feat_info.get('role') == 'INPUT':
                selected_features.append(feat)
        return selected_features
    
    def predict(self, df):
        return self.get_predictor().predict(df)
    
