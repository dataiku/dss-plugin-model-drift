#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import random
from collections import Counter
from datetime import datetime
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
EPOCH = datetime(1900, 1, 1)


class Preprocessor:

    def __init__ (self, df=None, target=None):
        self.df = df
        self.target = target
        self._categorical_features = []
        self._numerical_features = []
        self._text_features = []

    def check(self):
        if self.df is None:
            raise ValueError('df is not specified.')
        if self.target is None:
            raise ValueError('target is not specified.')

    def get_processed_train_test(self):
        self._categorical_features = [x for x in self._get_categorical_features() if x != self.target]
        self._numerical_features = self._get_numerical_features()
        self._text_features = self._get_text_features()
        self._parse_data()
        raw_train, raw_test = self._get_train_test_set()
        imputed_train, imputed_test = self._impute(raw_train, raw_test)
        dummy_values_dict = self._select_dummy_values(imputed_train, self._categorical_features)
        final_train = self._dummy_encode(imputed_train, dummy_values_dict)
        final_test = self._dummy_encode(imputed_test, dummy_values_dict)
        return final_train, final_test

    def _parse_data(self):
        def _datetime_to_epoch(series):
            return (series - EPOCH) / np.timedelta64(1, 's')

        for feature in self._categorical_features:
            self.df[feature] = self.df[feature].apply(self._coerce_to_unicode)
        for feature in self._text_features:
            self.df[feature] = self.df[feature].apply(self._coerce_to_unicode)
        for feature in self._numerical_features:
            if self.df[feature].dtype == np.dtype('M8[ns]'):
                self.df[feature] = _datetime_to_epoch(self.df[feature])
            else:
                self.df[feature] = self.df[feature].astype('double')

    def _get_numerical_features(self):
        return self.df.select_dtypes(include=['number']).columns.tolist()

    def _get_categorical_features(self):
        return self.df.select_dtypes(include=['object', 'category']).columns.tolist()

    def _get_text_features(self):
        return []

    def _coerce_to_unicode(self, x):
        if sys.version_info < (3, 0):
            if isinstance(x, str):
                return unicode(x, 'utf-8')
            else:
                return unicode(x)
        else:
            return str(x)

    def _select_dummy_values(self, dfx, features, LIMIT_DUMMIES = 100):
        dummy_values = {}
        for feature in features:
            values = [
                value
                for (value, _) in Counter(dfx[feature]).most_common(LIMIT_DUMMIES)
            ]
            dummy_values[feature] = values
        return dummy_values

    def _get_train_test_set(self, prop=0.8, seed=1234):
        k = int(self.df.shape[0] * prop)
        random.seed(seed)
        sampler = random.sample(self.df.index.tolist(), k)
        train = self.df.loc[sampler]
        test = self.df[~self.df.index.isin(sampler)]
        return train, test

    def _impute(self, df_train, df_test):
        for feature in self._numerical_features:
            v = df_train[feature].mean()
            df_train[feature] = df_train[feature].fillna(v)
            df_test[feature] = df_test[feature].fillna(v)
            logger.info('Imputed missing values in feature %s with value %s' % (feature, self._coerce_to_unicode(v)))

        for feature in self._categorical_features:
            v = 'NULL_CATEGORY'
            df_train[feature] = df_train[feature].fillna(v)
            df_test[feature] = df_test[feature].fillna(v)
            logger.info('Imputed missing values in feature %s with value %s' % (feature, self._coerce_to_unicode(v)))

        return df_train, df_test

    def _dummy_encode(self, dfx, dummy_values_dict):
        dfx_copy = dfx.copy()
        for (feature, dummy_values) in dummy_values_dict.items():
            for dummy_value in dummy_values:
                #TODO add dummy:N/A and dummy:_Others_
                dummy_name = u'dummy:%s:%s' % (feature, self._coerce_to_unicode(dummy_value))
                dfx_copy[dummy_name] = (dfx_copy[feature] == dummy_value).astype(float)
            del dfx_copy[feature]
            logger.info('Dummy-encoded feature %s' % feature)

        return dfx_copy
