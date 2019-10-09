# This is a test file intended to be used with pytest
# pytest automatically runs all the function starting with "test_"
# see https://docs.pytest.org for more information

import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
#from unittest.mock import Mock

RANDOM_SEED = 65537 # Fermat prime number <3

## Add stuff to the path to enable exec outside of DSS
plugin_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(plugin_root, 'python-lib'))

from dku_data_drift import DriftAnalyzer, ModelAccessor


def load_data():
    iris = load_iris()
    feature_names = iris['feature_names']
    target = 'target'
    df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                      columns=feature_names + [target])
    return df, feature_names, target


class ScikitPredictor:
    def __init__(self, df, feature_names, target):
        self.feature_names = feature_names
        self._clf = RandomForestClassifier(n_estimators=10, random_state=RANDOM_SEED).fit(df[feature_names], df[target])

    def get_features(self):
        return self.feature_names

    def predict(self, X):
        predictions = self._clf.predict(X[self.feature_names])
        probas =  self._clf.predict_proba(X[self.feature_names])
        df = pd.DataFrame(probas, columns = ['proba_{}'.format(x) for x in xrange(probas.shape[1])])
        df['prediction'] = predictions
        return df


class ScikitModelHandler:

    def __init__(self):
        self.df, self.feature_names, self.target = load_data()
        self.train_df = self.df.iloc[:100]
        self.test_df = self.df.iloc[:100] # same as train
        self.predictor = ScikitPredictor(self.train_df, self.feature_names, self.target)

    def get_predictor(self):
        return self.predictor

    def get_target_variable(self):
        return self.target

    def get_test_df(self):
        return [self.test_df, True]

    def get_per_feature(self):
        per_feature_dict ={
            self.target: {'role': 'TARGET'}
        }
        for feature in self.feature_names:
            dct = {
                'role': 'INPUT',
                'type': 'NUMERIC',
                'missing_handling': 'IMPUTE',
                'missing_impute_with': 'MEAN',
                'numerical_handling': 'REGULAR',
                'rescaling': 'AVGSTD',
            }
            per_feature_dict[feature] = dct

        return per_feature_dict

    def get_selected_features(self):
        selected_features = []
        for feat, feat_info in self.get_per_feature().items():
            if feat_info.get('role') == 'INPUT':
                selected_features.append(feat)
        return selected_features


class TestDriftAnalyzer:

    def setup(self):
        self.model_handler = 'model_handler'
        self.model_handler = ScikitModelHandler()
        self.model_accessor = ModelAccessor(self.model_handler)
        self.drifter = DriftAnalyzer(self.model_accessor)


    def test_identical_set(self):
        df, _, _ = load_data()
        new_test_df = df.iloc[:100] # same as in the training phase
        drift_features, drift_clf = self.drifter.train_drift_model(new_test_df)
        result_dict = self.drifter.compute_drift_metrics(new_test_df, drift_features, drift_clf)


        drift_accuracy = result_dict.get('drift_accuracy')
        kde = result_dict.get('kde')
        feature_importance = result_dict.get('feature_importance_metrics')
        stat_metrics = result_dict.get('stat_metrics')
        fugacity = result_dict.get('fugacity')
        label_list = result_dict.get('label_list')


        assert drift_accuracy == 0.5 # no drift, no difference, model can not distinguish
        for fugacity_one_class in fugacity:
            assert fugacity_one_class.get('New test set') == fugacity_one_class.get('Original test set')












