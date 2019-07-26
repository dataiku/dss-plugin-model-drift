# -*- coding: utf-8 -*-
import logging
import sys

from dataiku import pandasutils as pdu

logger = logging.getLogger(__name__)


class Preprocessor:

    def __init__(self, df, target='dku_flag'):
        self.df = df
        self.target = target
        self.categorical_features = []
        self.numerical_features = []
        self.text_features = []

    @staticmethod
    def _coerce_to_unicode(self, x):
        if sys.version_info < (3, 0):
            if isinstance(x, str):
                return unicode(x, 'utf-8')
            else:
                return unicode(x)
        else:
            return str(x)

    @staticmethod
    def _select_dummy_values(dfx, features, limit_dummies=100):
        from collections import Counter
        dummy_values = {}
        for feature in features:
            values = [
                value
                for (value, _) in Counter(dfx[feature]).most_common(limit_dummies)
            ]
            dummy_values[feature] = values
        return dummy_values

    def _get_numerical_features(self):
        return self.df.select_dtypes(include=['number']).columns.tolist()

    def _get_categorical_features(self):
        return self.df.select_dtypes(include=['object', 'category']).columns.tolist()

    def _get_text_features(self):
        return []

    def parse_data(self):
        from dataiku.doctor.utils import datetime_to_epoch
        for feature in self.categorical_features:
            self.df[feature] = self.df[feature].apply(self._coerce_to_unicode)
        for feature in self.text_features:
            self.df[feature] = self.df[feature].apply(self._coerce_to_unicode)
        for feature in self.numerical_features:
            if self.df[feature].dtype == np.dtype('M8[ns]'):
                self.df[feature] = datetime_to_epoch(self.df[feature])
            else:
                self.df[feature] = self.df[feature].astype('double')

    def _split_train_test(self):
        return pdu.split_train_valid(self.df, prop=0.8)

    def impute(self, dfx):
        for feature in self.numerical_features:
            v = dfx[feature].mean()
            dfx[feature] = dfx[feature].fillna(v)
            logger.info('Imputed missing values in feature %s with value %s' % (feature, self._coerce_to_unicode(v)))

        for feature in self.categorical_features:
            v = 'NULL_CATEGORY'
            dfx[feature] = dfx[feature].fillna(v)
            logger.info('Imputed missing values in feature %s with value %s' % (feature, self._coerce_to_unicode(v)))

        return dfx

    def dummy_encode(self, dfx, dummy_values_dict):
        dfx_copy = dfx.copy()
        for (feature, dummy_values) in dummy_values_dict.items():
            for dummy_value in dummy_values:
                dummy_name = u'%s_value_%s' % (feature, self._coerce_to_unicode(dummy_value))
                dfx_copy[dummy_name] = (dfx_copy[feature] == dummy_value).astype(float)
            del dfx_copy[feature]
            logger.info('Dummy-encoded feature %s' % feature)

        return dfx_copy

    def get_processed_train_test(self):

        self.categorical_features = [x for x in self._get_categorical_features() if x != self.target]
        self.numerical_features = self._get_numerical_features()
        self.text_features = self._get_text_features()
        self.parse_data()
        raw_train, raw_test = self._split_train_test()
        imputed_train = self.impute(raw_train)
        imputed_test = self.impute(raw_test)

        dummy_values_dict = self._select_dummy_values(imputed_train, self.categorical_features)
        final_train = self.dummy_encode(imputed_train, dummy_values_dict)
        final_test = self.dummy_encode(imputed_test, dummy_values_dict)

        return final_train, final_test