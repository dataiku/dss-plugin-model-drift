import dataiku
import pandas as pd, numpy as np
from dataiku.customrecipe import *

from dku_data_drift.drift_analyzer import DriftAnalyzer
from dku_data_drift.dataframe_helpers import schema_are_compatible
from dku_data_drift.dataset_helpers import get_partitioning_columns

import datetime
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Model Drift Recipe | %(levelname)s - %(message)s')

# Retrieve reference dataset from input
input_names = get_input_names_for_role('original')
original_df = dataiku.Dataset(input_names[0]).get_dataframe(limit=100000)

# Retrieve the new dataset from input
input_names = get_input_names_for_role('new')
new_df = dataiku.Dataset(input_names[0]).get_dataframe(limit=100000)

# Retrieve the output dataset
output_names = get_output_names_for_role('main_output')
output_datasets = [dataiku.Dataset(name) for name in output_names]
output_dataset = output_datasets[0]

# Retrieve the learning task
learning_task = get_recipe_config().get('learning_task')
if learning_task is None:
    raise ValueError('Learning task must be defined.')

# Handle partitioning
partition_cols_new_df = get_partitioning_columns(new_df)
partition_cols_original_df = get_partitioning_columns(original_df)
new_df = new_df.drop(partition_cols_new_df, axis=1)
original_df = original_df.drop(partition_cols_original_df, axis=1)
if len(list(new_df.columns))==0 or len(list(original_df.columns))==0:
    raise ValueError('Without the partition column, at least one of the datasets is empty.')

# Analyse the drift
drifter = DriftAnalyzer(learning_task)
drifter.fit(new_df=new_df, original_df=original_df)

# Write metrics and drift score in output dataset
timestamp = datetime.datetime.now()
drift_score = drifter.get_drift_score()

metrics_row = {'timestamp': [timestamp], 'drift_score': [drift_score]}
new_df = pd.DataFrame(metrics_row, columns=['timestamp', 'drift_score'])

if output_dataset.cols is None:
    logger.info("Dataset is empty, writing the new metrics in a new table")
    output_dataset.write_with_schema(new_df)
else:
    logger.info("Dataset is not empty, append the new metrics to the previous table")
    existing_df = output_dataset.get_dataframe()
    if not schema_are_compatible(existing_df, new_df):
        raise ValueError('Schema are not equal, concatenation is not possible.')
    concatenate_df = pd.concat([existing_df, new_df], axis=0)
    concatenate_df.columns = ['timestamp', 'model_id', 'model_version', 'drift_score']
    output_dataset.write_with_schema(concatenate_df)

