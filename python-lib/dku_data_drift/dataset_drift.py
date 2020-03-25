"""
The tools and class for the analysis of datasets drift. Differs from the drift_analyzer.py classes
because there is no model and model accessor. The class of this analysis takes only datasets into account.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class DataDriftAnalyzer:

    def __init__(self, original_df, new_df):
        """
        :param: original_df: The original dataset for the analysis of the drift
            :type: Pandas Dataframe

        :param: new_df: The dataset we want to compare with the original distribution
            :type: Same
        """
        self.original_df = original_df
        self.new_df = new_df
        self.merged_df = None
        self.labels_origin = None
        self.clf = None

    def build_merged_dataset(self):
        """
        Prepare the labels of both datasets and build the mixed origin dataset for drift detection
        """
        if not len(self.original_df) or not len(self.new_df):
            raise ValueError("A dataframe has not been defined. Check that the two dataframes are received.")
        y_original = np.ones(len(self.original_df))
        y_new = np.zeros(len(self.new_df))
        self.labels_origin = np.concatenate([y_original, y_new])
        del y_original, y_new
        self.merged_df = pd.concat([self.original_df, self.new_df])

    def train_drift_analyser(self):
        """
        Train a classifier to detect the drift on two datasets
        """
        X_train, X_test, y_train, y_test = train_test_split(self.merged_df,
                                                            self.labels_origin,
                                                            test_size = 0.33,
                                                            random_state = 42)
        clf = RandomForestClassifier(n_estimators=100, random_state=1337, max_depth=13, min_samples_leaf=1)
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        drift_score = np.mean(y_predict == y_test)
        return drift_score
