#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    Simple functions helpers
"""

import logging
import numpy as np
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Model Drift Plugin | %(levelname)s - %(message)s')

logger.info("Python version: {}".format(sys.version))
# python3 does not have basetring
try:
    basestring
except NameError:
    basestring = str


def not_enough_data(df, min_len=1):
    """
        Compare length of dataframe to minimum lenght of the test data.
        Used in the relevance of the measure.
    :param df: Input dataframe
    :param min_len:
    :return:
    """
    return len(df) < min_len


def nothing_to_do(stuff):
    return stuff is None


def generic_check_compute_arguments(datetime_column, groupby_columns):
    """
        Check columns argument in the dataframe. Date is always tricky to handle.
    :param datetime_column:
    :param groupby_columns:
    :return:
    """
    if not isinstance(datetime_column, basestring):
        raise ValueError('datetime_column param must be string. Got: ' + str(datetime_column))
    if groupby_columns:
        if not isinstance(groupby_columns, list):
            raise ValueError('groupby_columns param must be an array of strings. Got: ' + str(groupby_columns))
        for col in groupby_columns:
            if not isinstance(col, basestring):
                raise ValueError('groupby_columns param must be an array of strings. Got: ' + str(col))