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


def get_windowing_params(recipe_config):
    def _p(param_name, default=None):
        return recipe_config.get(param_name, default)

    causal_window = _p('causal_window')
    window_unit = _p('window_unit')
    window_width = int(_p('window_width'))
    if _p('window_type') == 'none':
        window_type = None
    else:
        window_type = _p('window_type')

    if window_type == 'gaussian':
        gaussian_std = _p('gaussian_std')
    else:
        gaussian_std = None

    closed_option = _p('closed_option')
    aggregation_types = _p('aggregation_types')

    params = WindowAggregatorParams(window_unit=window_unit,
                                window_width=window_width,
                                window_type=window_type,
                                gaussian_std=gaussian_std,
                                closed_option=closed_option,
                                causal_window=causal_window,
                                aggregation_types=aggregation_types)

    params.check()
    return params


def get_interval_restriction_params(recipe_config):
    def _p(param_name, default=None):
        return recipe_config.get(param_name, default)

    min_valid_values_duration_value = _p('min_valid_values_duration_value')
    min_deviation_duration_value = _p('min_deviation_duration_value')
    time_unit = _p('time_unit')

    params = IntervalRestrictorParams(min_valid_values_duration_value=min_valid_values_duration_value,
                                      max_deviation_duration_value=min_deviation_duration_value,
                                      time_unit=time_unit)

    params.check()
    return params


def get_extrema_extraction_params(recipe_config):
    def _p(param_name, default=None):
        return recipe_config.get(param_name, default)

    causal_window = _p('causal_window')
    window_unit = _p('window_unit')
    window_width = int(_p('window_width'))
    if _p('window_type') == 'none':
        window_type = None
    else:
        window_type = _p('window_type')

    if window_type == 'gaussian':
        gaussian_std = _p('gaussian_std')
    else:
        gaussian_std = None
    closed_option = _p('closed_option')
    extrema_type = _p('extrema_type')
    aggregation_types = _p('aggregation_types')

    print('AGGREGATION TYPE: ', aggregation_types)
    window_params = WindowAggregatorParams(window_unit=window_unit,
                                       window_width=window_width,
                                       window_type=window_type,
                                       gaussian_std=gaussian_std,
                                       closed_option=closed_option,
                                       causal_window=causal_window,
                                       aggregation_types=aggregation_types)

    window_aggregator = WindowAggregator(window_params)
    params = ExtremaExtractorParams(window_aggregator=window_aggregator, extrema_type=extrema_type)
    params.check()
    return params
