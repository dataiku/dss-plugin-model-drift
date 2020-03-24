# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from dku_data_drift.model_tools import SurrogateModel
import logging

logger = logging.getLogger(__name__)

ALGORITHMS_WITH_VARIABLE_IMPORTANCE = [RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, DecisionTreeClassifier]
MAX_NUM_ROW = 100000

class ModelAccessor:
    def __init__(self, model_handler=None):
        self.model_handler = model_handler

    def get_prediction_type(self):
        if 'CLASSIFICATION' in self.model_handler.get_prediction_type():
            return 'CLASSIFICATION'
        elif 'REGRESSION' in self.model_handler.get_prediction_type():
            return 'REGRESSION'
        else:
            return 'CLUSTERING'

    def check(self):
        if self.model_handler is None:
            raise ValueError('model_handler object is not specified')
            
    def get_target_variable(self):
        return self.model_handler.get_target_variable()

    def get_original_test_df(self, limit=MAX_NUM_ROW):
        try:
            return self.model_handler.get_test_df()[0][:limit]
        except Exception as e:
            logger.warning('Can not retrieve original test set: {}. The plugin will take the whole original dataset.'.format(e))
            return self.model_handler.get_full_df()[0][:limit]

    def get_per_feature(self):
        return self.model_handler.get_per_feature()

    def get_predictor(self):
        return self.model_handler.get_predictor()

    def get_feature_importance(self, cumulative_percentage_threshold=95):
        """
        :param cumulative_percentage_threshold:
        :return:
        """
        if self._algorithm_is_tree_based():
            predictor = self.get_predictor()
            clf = predictor._clf
            feature_importance = []
            feature_importances = clf.feature_importances_
            feature_names = predictor.get_features()
            for feature_name, feat_importance in zip(feature_names, feature_importances):
                feature_importance.append({
                    'feature': feature_name,
                    'importance': 100*feat_importance/sum(feature_importances)
                })

            dfx = pd.DataFrame(feature_importance).sort_values(by='importance', ascending=False).reset_index(drop=True)
            dfx['cumulative_importance'] = dfx['importance'].cumsum()
            dfx_top = dfx.loc[dfx['cumulative_importance'] <= cumulative_percentage_threshold]
            return dfx_top.rename_axis('rank').reset_index().set_index('feature')
        else: # use surrogate model
            logger.info('Fitting surrogate model ...')
            surrogate_model = SurrogateModel(self.get_prediction_type())
            original_test_df = self.get_original_test_df()
            predictions_on_original_test_df = self.get_predictor().predict(original_test_df)
            surrogate_target = 'dku_predicted_label'
            surrogate_df = original_test_df[self.get_selected_features()]
            surrogate_df[surrogate_target] = predictions_on_original_test_df['prediction']
            surrogate_model.fit(surrogate_df, surrogate_target)
            return surrogate_model.get_feature_importance()


    def get_selected_features(self):
        selected_features = []
        for feat, feat_info in self.get_per_feature().items():
            if feat_info.get('role') == 'INPUT':
                selected_features.append(feat)
        return selected_features

    def predict(self, df):
        return self.get_predictor().predict(df)


    def _algorithm_is_tree_based(self):
        predictor = self.get_predictor()
        algo = predictor._clf
        for algorithm in ALGORITHMS_WITH_VARIABLE_IMPORTANCE:
            if isinstance(algo, algorithm):
                return True
            elif predictor.params.modeling_params.get('algorithm') == 'XGBOOST_CLASSIFICATION':
                return True
        return False