import dataiku
import pandas as pd
import sys
import json
from dataiku.customrecipe import *
from dku_tools import set_column_description
from dku_data_drift.drift_analyzer import DriftAnalyzer
from dku_data_drift.dataframe_helpers import schema_are_compatible
from dku_data_drift.dataset_helpers import get_partitioning_columns

import datetime
import logging


MAX_NUM_ROW = 100000

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Model Drift Recipe | %(levelname)s - %(message)s')

# Retrieve the reference datase
input_names = get_input_names_for_role('original')
original_ds = dataiku.Dataset(input_names[0])
original_df = original_ds.get_dataframe(limit=MAX_NUM_ROW)

# Retrieve the new dataset
input_names = get_input_names_for_role('new')
new_ds = dataiku.Dataset(input_names[0])
new_df = new_ds.get_dataframe(limit=MAX_NUM_ROW)

# Retrieve the output dataset where the metrics and drift score will be written
output_names = get_output_names_for_role('main_output')
output_datasets = [dataiku.Dataset(name, ignore_flow=True) for name in output_names]
output_dataset = output_datasets[0]

# Retrieve the learning task
learning_task = get_recipe_config().get('learning_task')
if learning_task is None:
    raise ValueError('Learning task must be defined.')

# Retrieve the target variable
target_variable = get_recipe_config().get('target_variable')
if target_variable is None:
    raise ValueError('Target variable must be defined.')

metric_list = get_recipe_config().get('metric_list')
if len(metric_list) == 0 or metric_list is None:
    raise ValueError('Please choose at least one metric.')
logger.info('Chosen metrics: ', metric_list)


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
    fugacity = drifter.get_fugacity()
    for k,v in fugacity.items():
        new_df[k] = [v]
        column_description_dict[k] = 'The difference between the ratio percentage of this class in the new dataset compared to that in the original dataset. Positive means there is an increase and vice versa'

if 'feature_importance' in metric_list:
    feature_importance = drifter.get_drift_feature_importance()
    feat_dict = {}
    for feat, feat_info in feature_importance[:10].iterrows():
        feat_dict[feat] = round(feat_info.get('importance'), 2)
    new_df['drift_feature_importance'] = [json.dumps(feat_dict)]
    column_description_dict['drift_feature_importance'] = 'List of features that have been drifted the most, with their % of importance'

try:
    existing_df = output_dataset.get_dataframe()
    if schema_are_compatible(existing_df, new_df):
        logger.info("Dataset is not empty, append the new metrics to the previous table")
        concatenate_df = pd.concat([existing_df, new_df], axis=0)
        columns_order = ['timestamp'] + [col for col in concatenate_df.columns if col != 'timestamp']
        concatenate_df = concatenate_df[columns_order]
        output_dataset.write_with_schema(concatenate_df)
    else:
        logger.info("Schema not compatible, overwriting the metric table.")
        columns_order = ['timestamp'] + [col for col in new_df.columns if col != 'timestamp']
        new_df = new_df[columns_order]
        output_dataset.write_with_schema(new_df)

except Exception as e:
    from future.utils import raise_
    if "No column in schema" in str(e) or 'No JSON object could be decoded' in str(e):
        logger.info("Dataset is empty, writing the new metrics in a new table")
        output_dataset.write_with_schema(new_df)
    else:
        raise_(Exception, "Fail to write to dataset: {}".format(e), sys.exc_info()[2])


set_column_description(output_dataset, column_description_dict)