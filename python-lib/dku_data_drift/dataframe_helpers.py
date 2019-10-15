# coding: utf-8
import numpy as np


# python3 does not have basetring
try:
    basestring
except NameError:
    basestring = str


def not_enough_data(df, min_len=1):
    return len(df) < min_len

def nothing_to_do(stuff):
    return stuff is None


def generic_check_compute_arguments(datetime_column, groupby_columns):
    if not isinstance(datetime_column, basestring):
        raise ValueError('datetime_column param must be string. Got: ' + str(datetime_column))
    if groupby_columns:
        if not isinstance(groupby_columns, list):
            raise ValueError('groupby_columns param must be an array of strings. Got: ' + str(groupby_columns))
        for col in groupby_columns:
            if not isinstance(col, basestring):
                raise ValueError('groupby_columns param must be an array of strings. Got: ' + str(col))