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
original_df = original_ds.get_dataframe(limit=MAX_NUM_ROW)

# Retrieve the new dataset from input
input_names = get_input_names_for_role('new')
new_ds = dataiku.Dataset(input_names[0])
new_df = new_ds.get_dataframe(limit=MAX_NUM_ROW)

# Retrieve the output dataset
output_names = get_output_names_for_role('main_output')
output_datasets = [dataiku.Dataset(name) for name in output_names]
output_dataset = output_datasets[0]


target_variable = None
learning_task = None
output_format = 'single_column'
prediction_column_available = get_recipe_config().get('prediction_column_available')
if prediction_column_available:
    metric_list = get_recipe_config().get('metric_list_with_prediction')
    learning_task = get_recipe_config().get('learning_task')
    if learning_task is None:
        raise ValueError('Learning task must be defined.')
    target_variable = get_recipe_config().get('target_variable')
    if target_variable is None:
        raise ValueError('Target variable must be defined.')
    output_format = get_recipe_config().get('output_format', 'single_column')

else:
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

if 'fugacity' in metric_list:
    if drifter.get_prediction_type() == 'CLASSIFICATION':
        fugacity = drifter.get_classification_fugacity()
        if output_format == 'multiple_columns':
            for k,v in fugacity.items():
                new_df[k] = [v]
                column_description_dict[k] = 'The difference between the ratio percentage of this class in the new dataset compared to that in the original dataset. Positive means there is an increase and vice versa'
        else:
            new_df['fugacity'] = json.dumps(fugacity)
            column_description_dict['fugacity'] = 'The difference between the ratio percentage of a class in the new dataset compared to that in the original dataset. Positive means there is an increase and vice versa'
    else: # regression
        fugacity, bin_description = drifter.get_regression_fugacity()
        if output_format == 'multiple_columns':
            for k, v in enumerate(fugacity.items()):
                new_df[v[0]] = [v[1].values[0]]
                column_description_dict[v[0]] = bin_description[k]
        else:
            new_df['fugacity'] = json.dumps(fugacity.iloc[0].to_dict())
            proper_bin_description = ', '.join(
                ['bin {0}: {1}'.format(bin_index, bin_desc) for bin_index, bin_desc in enumerate(bin_description)])
            column_description_dict['fugacity'] = proper_bin_description


if 'feature_importance' in metric_list:
    feature_importance = drifter.get_drift_feature_importance()
    feat_dict = {}
    for feat, feat_info in feature_importance[:10].iterrows():
        feat_dict[feat] = round(feat_info.get('importance'), 2)
    new_df['drift_feature_importance'] = [json.dumps(feat_dict)]
    column_description_dict['drift_feature_importance'] = 'List of features that have been drifted the most, with their % of importance'

output_dataset.write_with_schema(new_df)
set_column_description(output_dataset, column_description_dict)

