import dataiku
import pandas as pd
import sys
from dataiku.customrecipe import *
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

# Retrieve the target variable
target_variable = get_recipe_config().get('target_variable')
if target_variable is None:
    raise ValueError('Target variable must be defined.')

# Retrieve the output dataset where the metrics and drift score will be written
output_names = get_output_names_for_role('main_output')
output_datasets = [dataiku.Dataset(name, ignore_flow=True) for name in output_names]
output_dataset = output_datasets[0]

# Retrieve the learning task
learning_task = get_recipe_config().get('learning_task')
if learning_task is None:
    raise ValueError('Learning task must be defined.')

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
drift_score = drifter.get_drift_score()

metrics_row = {'timestamp': [timestamp], 'drift_score': [drift_score]}
new_df = pd.DataFrame(metrics_row, columns=['timestamp', 'drift_score'])


try:
    existing_df = output_dataset.get_dataframe()
    if not schema_are_compatible(existing_df, new_df):
        raise ValueError('Schema are not equal, concatenation is not possible.')
    logger.info("Dataset is not empty, append the new metrics to the previous table")
    concatenate_df = pd.concat([existing_df, new_df], axis=0)
    columns_order = ['timestamp', 'drift_score']
    concatenate_df = concatenate_df[columns_order]
    output_dataset.write_with_schema(concatenate_df)
except Exception as e:
    from future.utils import raise_
    if "No column in schema" in str(e) or 'No JSON object could be decoded' in str(e):
        logger.info("Dataset is empty, writing the new metrics in a new table")
        output_dataset.write_with_schema(new_df)
    else:
        raise_(Exception, "Fail to write to dataset: {}".format(e), sys.exc_info()[2])