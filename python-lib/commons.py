# coding: utf-8
import dataiku
from dataiku.customrecipe import *

from dku_timeseries import ResamplerParams
from dku_timeseries import WindowAggregator
from dku_timeseries import WindowAggregatorParams
from dku_timeseries import IntervalRestrictorParams
from dku_timeseries import ExtremaExtractorParams

def get_input_output():
    if len(get_input_names_for_role('input_dataset')) == 0:
        raise ValueError('No input dataset.')
    input_dataset_name = get_input_names_for_role('input_dataset')[0]
    input_dataset = dataiku.Dataset(input_dataset_name)
    output_dataset_name = get_output_names_for_role('output_dataset')[0]
    output_dataset = dataiku.Dataset(output_dataset_name)
    return (input_dataset, output_dataset)

def get_resampling_params(recipe_config):
    def _p(param_name, default=None):
        return recipe_config.get(param_name, default)

    interpolation_method = _p('interpolation_method')
    extrapolation_method = _p('extrapolation_method')
    time_step = _p('time_step')
    time_unit = _p('time_unit')
    clip_start = _p('clip_start')
    clip_end = _p('clip_end')
    shift = _p('shift')

    params = ResamplerParams(interpolation_method=interpolation_method,
                             extrapolation_method=extrapolation_method,
                             time_step=time_step,
                             time_unit=time_unit,
                             clip_start=clip_start,
                             clip_end=clip_end,
                             shift=shift)
    params.check()
    return params
