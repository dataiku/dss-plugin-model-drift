# This is a test file intended to be used with pytest
# pytest automatically runs all the function starting with "test_"
# see https://docs.pytest.org for more information

import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import pytest
## Add stuff to the path to enable exec outside of DSS
plugin_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(plugin_root, 'python-lib'))


from dku_data_drift import DriftAnalyzer, ModelAccessor



RANDOM_SEED = 65537 # Fermat prime number <3
TEST_RATIO = 0.3 # change this will affect model's prediction result

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
        self.train_df, self.test_df = train_test_split(self.df, test_size=0.3, random_state=RANDOM_SEED)
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

    def test_empty_set(self):
        _, feature_names, _ = load_data()
        new_test_df = pd.DataFrame(columns=feature_names)
        with pytest.raises(Exception) as e_info:
            drift_features, drift_clf = self.drifter.train_drift_model(new_test_df)

    def test_one_row_set(self):
        _, feature_names, _ = load_data()
        new_test_df = pd.DataFrame(columns=feature_names)
        with pytest.raises(Exception) as e_info:
            drift_features, drift_clf = self.drifter.train_drift_model(new_test_df)

    def test_missing_feature_set(self):
        df, feature_names, _ = load_data()
        _, new_test_df = train_test_split(df, test_size=TEST_RATIO, random_state=RANDOM_SEED)
        new_test_df = new_test_df.drop(feature_names[0], 1)

        with pytest.raises(Exception) as e_info:
            drift_features, drift_clf = self.drifter.train_drift_model(new_test_df, min_num_row=10)

    def test_identical_set(self):
        df, _, _ = load_data()
        _, new_test_df = train_test_split(df, test_size=TEST_RATIO, random_state=RANDOM_SEED)
        drift_features, drift_clf = self.drifter.train_drift_model(new_test_df, min_num_row=10)
        result_dict = self.drifter.compute_drift_metrics(drift_features, drift_clf)

        drift_accuracy = result_dict.get('drift_accuracy')
        fugacity = result_dict.get('fugacity')
        feature_importance = result_dict.get('feature_importance')

        original_model_feature_importance = sorted([feat_imp['original_model'] for feat_imp in feature_importance])
        drift_model_feature_importance = sorted([feat_imp['drift_model'] for feat_imp in feature_importance])

        assert drift_accuracy == 0.01 # no drift, model can not distinguish, accuracy is 0.5 -> transformed score is 0.01
        for fugacity_one_class in fugacity:
            assert fugacity_one_class.get('Selected dataset') == fugacity_one_class.get('Original dataset')

        assert np.array_equal(original_model_feature_importance, sorted([0.01, 46.77454270154651, 43.17215785326303, 0.01]))
        assert np.array_equal(drift_model_feature_importance, sorted([24.828325796836367, 27.141709051012583, 0.01, 26.639325904270546]))


    def test_drifted_set(self):

        df, feature_names, _ = load_data()
        _, original_test_df = train_test_split(df, test_size=TEST_RATIO, random_state=RANDOM_SEED)
        new_test_df = original_test_df.copy()
        new_test_df[feature_names] = new_test_df[feature_names] * 2 # shift the feature distribution

        drift_features, drift_clf = self.drifter.train_drift_model(new_test_df, min_num_row=10)
        result_dict = self.drifter.compute_drift_metrics(drift_features, drift_clf)

        drift_accuracy = result_dict.get('drift_accuracy')
        fugacity = result_dict.get('fugacity')

        prediction_distribution_original_test_set = [fuga['Original dataset'] for fuga in fugacity]
        prediction_distribution_new_test_set = [fuga['Selected dataset'] for fuga in fugacity]


        assert drift_accuracy == 1 
        assert np.array_equal(prediction_distribution_original_test_set, [24.44, 40.0, 35.56])
        assert np.array_equal(prediction_distribution_new_test_set, [22.22, 2.22, 75.56])









