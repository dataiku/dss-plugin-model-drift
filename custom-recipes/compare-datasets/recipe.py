import dataiku
import pandas as pd
import json
from dku_tools import set_column_description
from dataiku.customrecipe import *
from dku_data_drift.drift_analyzer import DriftAnalyzer
from dku_data_drift.dataset_helpers import get_partitioning_columns

import datetime
import logging

MAX_NUM_ROW = 100000 # heuristic choice

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Model Drift Recipe | %(levelname)s - %(message)s')

# Retrieve reference dataset from input
input_names = get_input_names_for_role('original')
original_ds = dataiku.Dataset(input_names[0])
original_df = original_ds.get_dataframe(bool_as_str=True, limit=MAX_NUM_ROW)

# Retrieve the new dataset from input
input_names = get_input_names_for_role('new')
new_ds = dataiku.Dataset(input_names[0])
new_df = new_ds.get_dataframe(bool_as_str=True, limit=MAX_NUM_ROW)

# Retrieve the output dataset
output_names = get_output_names_for_role('main_output')
output_datasets = [dataiku.Dataset(name) for name in output_names]
output_dataset = output_datasets[0]

target_variable = None
learning_task = None
output_format = 'single_column'
metric_list = get_recipe_config().get('metric_list_without_prediction')

if len(metric_list) == 0 or metric_list is None:
    raise ValueError('Please choose at least one metric.')
logger.info('Chosen metrics: ', metric_list)

# Handle partitioning
partition_cols_new_df = get_partitioning_columns(new_ds)
partition_cols_original_df = get_partitioning_columns(original_ds)
if partition_cols_original_df:
    original_df = original_df.drop(partition_cols_original_df, axis=1)
if partition_cols_new_df:
    new_df = new_df.drop(partition_cols_new_df, axis=1)
if len(new_df.columns)==0 or len(original_df.columns)==0:
    raise ValueError('Without the partition column, at least one of the datasets is empty.')

# Handle columns to remove
columns_to_remove = get_recipe_config().get('columns_to_remove')
if len(columns_to_remove) != 0:
    to_remove_in_original = set(original_df.columns).intersection(set(columns_to_remove))
    if to_remove_in_original:
        original_df = original_df.drop(list(to_remove_in_original), axis=1)
    to_remove_in_new = set(new_df.columns).intersection(set(columns_to_remove))
    if to_remove_in_new:
        new_df = new_df.drop(list(to_remove_in_new), axis=1)

# Analyse the drift
drifter = DriftAnalyzer(learning_task)
drifter.fit(new_df=new_df, original_df=original_df, target=target_variable)

# Write the drift score and metrics
timestamp = datetime.datetime.now()
new_df = pd.DataFrame({'timestamp': [timestamp]})

column_description_dict = {}

if 'drift_score' in metric_list:
    drift_score = drifter.get_drift_score()
    new_df['drift_score'] = [drift_score]
    column_description_dict['drift_score'] = 'The drift score (between 0 and 1) is low if the new dataset and the original dataset are indistinguishable.'

if 'feature_importance' in metric_list:
    feature_importance = drifter.get_drift_feature_importance()
    feat_dict = {}
    for feat, feat_info in feature_importance[:10].iterrows():
        feat_dict[feat] = round(feat_info.get('importance'), 2)
    new_df['most_drifted_features'] = [json.dumps(feat_dict)]
    column_description_dict['most_drifted_features'] = 'List of features that have been drifted the most, with their % of importance'


output_dataset.write_with_schema(new_df)
set_column_description(output_dataset, column_description_dict)

