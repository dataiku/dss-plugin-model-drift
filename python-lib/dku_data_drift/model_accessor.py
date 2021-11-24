# -*- coding: utf-8 -*-
import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from dku_data_drift.model_tools import SurrogateModel
from dku_data_drift.model_drift_constants import ModelDriftConstants

logger = logging.getLogger(__name__)

ALGORITHMS_WITH_VARIABLE_IMPORTANCE = [RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, DecisionTreeClassifier,
                                       RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, DecisionTreeRegressor]


class ModelAccessor(object):
    def __init__(self, model_handler=None):
        self.model_handler = model_handler

    def get_prediction_type(self):
        """
        Wrap the prediction type accessor of the model
        """
        if self.model_handler.get_prediction_type() in ModelDriftConstants.DKU_CLASSIFICATION_TYPE:
            return ModelDriftConstants.CLASSIFICATION_TYPE
        elif ModelDriftConstants.REGRRSSION_TYPE in self.model_handler.get_prediction_type():
            return ModelDriftConstants.REGRRSSION_TYPE
        else:
            return ModelDriftConstants.CLUSTERING_TYPE
            
    def get_target_variable(self):
        """
        Return the name of the target variable
        """
        return self.model_handler.get_target_variable()

    def get_original_test_df(self, limit=ModelDriftConstants.MAX_NUM_ROW):
        try:
            return self.model_handler.get_test_df()[0][:limit]
        except Exception as e:
            logger.warning('Can not retrieve original test set: {}. The plugin will take the whole original dataset.'.format(e))
            return self.model_handler.get_full_df()[0][:limit]

    def get_per_feature(self):
        return self.model_handler.get_per_feature()

    def get_predictor(self):
        return self.model_handler.get_predictor()

    def get_feature_importance(self, cumulative_percentage_threshold=ModelDriftConstants.FEAT_IMP_CUMULATIVE_PERCENTAGE_THRESHOLD):
        """
        :param cumulative_percentage_threshold: only return the top n features whose sum of importance reaches this threshold
        :return:
        """
        if self._algorithm_is_tree_based():
            predictor = self.get_predictor()
            clf = predictor._clf
            feature_names = predictor.get_features()
            feature_importances = clf.feature_importances_

        else: # use surrogate model
            logger.info('Fitting surrogate model ...')
            surrogate_model = SurrogateModel(self.get_prediction_type())
            original_test_df = self.get_original_test_df()
            predictions_on_original_test_df = self.get_predictor().predict(original_test_df)
            surrogate_df = original_test_df[self.get_selected_features()]
            surrogate_df[ModelDriftConstants.SURROGATE_TARGET] = predictions_on_original_test_df['prediction']
            surrogate_model.fit(surrogate_df, ModelDriftConstants.SURROGATE_TARGET)
            feature_names = surrogate_model.get_features()
            feature_importances = surrogate_model.clf.feature_importances_

        feature_importance = []
        for feature_name, feat_importance in zip(feature_names, feature_importances):
            feature_importance.append({
                ModelDriftConstants.FEATURE: feature_name,
                ModelDriftConstants.IMPORTANCE: 100 * feat_importance / sum(feature_importances)
            })

        dfx = pd.DataFrame(feature_importance).sort_values(by=ModelDriftConstants.IMPORTANCE, ascending=False).reset_index(drop=True)
        dfx[ModelDriftConstants.CUMULATIVE_IMPORTANCE] = dfx[ModelDriftConstants.IMPORTANCE].cumsum()
        dfx_top = dfx.loc[dfx[ModelDriftConstants.CUMULATIVE_IMPORTANCE] <= cumulative_percentage_threshold]
        return dfx_top.rename_axis(ModelDriftConstants.RANK).reset_index().set_index(ModelDriftConstants.FEATURE)


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
            elif predictor.params.modeling_params.get('algorithm') == 'XGBOOST_REGRESSION':
                return True
        return False