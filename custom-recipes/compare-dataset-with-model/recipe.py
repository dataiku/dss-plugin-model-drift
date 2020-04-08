import dataiku
import pandas as pd
import json
import sys
from dataiku.customrecipe import *
from dku_tools import set_column_description, get_train_date
from dku_data_drift.drift_analyzer import DriftAnalyzer
from dku_data_drift.model_accessor import ModelAccessor
from dku_data_drift.dataframe_helpers import schema_are_compatible
from dku_data_drift.dataset_helpers import get_partitioning_columns

from model_metadata import get_model_handler

import datetime
import logging

MAX_NUM_ROW = 100000 # heuristic choice

# init logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Model Drift Recipe | %(levelname)s - %(message)s')

client = dataiku.api_client()
project = client.get_project(dataiku.default_project_key())

# Retrieve input dataset
logger.info("Retrieve the input dataset")
input_names = get_input_names_for_role('input')
new_ds = dataiku.Dataset(input_names[0])
new_df = new_ds.get_dataframe(limit=MAX_NUM_ROW)

partition_cols_new_df = get_partitioning_columns(new_ds)
if partition_cols_new_df:
    new_df = new_df.drop(partition_cols_new_df, axis=1)
if len(new_df.columns)==0:
    raise ValueError('Without the partition column, dataset is empty.')

# Retrieve input model
logger.info("Retrieve the input model")
input_names = get_input_names_for_role('model')
model = dataiku.Model(input_names[0])
model_id = model.get_id()
saved_model = project.get_saved_model(model_id)
available_model_version = saved_model.list_versions()
logger.info("The input model has the following version ids: {}".format(available_model_version))

# Retrieve the version id of the model (dynamic dropdown selection)
version_id = get_recipe_config().get('version_id')
if version_id is None:
    raise ValueError('Please choose a model version.')

# Retrieve the output dataset for metrics and score
output_names = get_output_names_for_role('main_output')
output_datasets = [dataiku.Dataset(name, ignore_flow=True) for name in output_names]
output_dataset = output_datasets[0]

# Access the model
model_handler = get_model_handler(model=model, version_id=version_id)
model_accessor = ModelAccessor(model_handler)

# Analyze the drift
drifter = DriftAnalyzer(prediction_type=None)
target = model_accessor.get_target_variable()
drifter.fit(new_df, model_accessor=model_accessor)

if drifter.get_prediction_type() == 'CLASSIFICATION':
    metric_list = ['drift_score', 'fugacity', 'feature_importance']
elif drifter.get_prediction_type() == 'REGRESSION':
    metric_list = ['drift_score', 'feature_importance']
else:
    metric_list = ['drift_score']


# Write the drift score and metrics

model_train_date = get_train_date(model_id, version_id)

timestamp = datetime.datetime.now()
new_df = pd.DataFrame({'timestamp': [timestamp],
                       'model_id': [model_id],
                       'version_id': [version_id],
                       'train_date': [model_train_date]})

column_description_dict = {}

if 'drift_score' in metric_list:
    drift_score = drifter.get_drift_score()
    new_df['drift_score'] = [drift_score]
    column_description_dict['drift_score'] = 'The drift score (between 0 and 1) is low if the new dataset and the original dataset are indistinguishable.'

if 'fugacity' in metric_list:
    fugacity = drifter.get_fugacity()
    for k,v in fugacity.items():
        new_df[k] = [v]
        column_description_dict[k] = '{} is the difference between the ratio percentage of this class in the new dataset compared to that in the original dataset. Positive means there is an increase and vice versa'.format(k)

if 'feature_importance' in metric_list:
    drift_feature_importance = drifter.get_drift_feature_importance()
    feat_dict = {}
    for feat, feat_info in drift_feature_importance[:10].iterrows():
        feat_dict[feat] = round(feat_info.get('importance'), 2)
    new_df['drift_feature_importance'] = [json.dumps(feat_dict)]
    column_description_dict['drift_feature_importance'] = 'List of features that have been drifted the most, with their % of importance'

    original_feature_importance = drifter.get_original_feature_importance()
    feat_dict = {}
    for feat, feat_info in original_feature_importance[:10].iterrows():
        feat_dict[feat] = round(feat_info.get('importance'), 2)
    new_df['original_feature_importance'] = [json.dumps(feat_dict)]
    column_description_dict['original_feature_importance'] = 'List of the most important features in the deployed model, with their % of importance'

try:
    existing_df = output_dataset.get_dataframe()
    if schema_are_compatible(existing_df, new_df):
        logger.info("Dataset is not empty, append the new metrics to the previous table")
        concatenate_df = pd.concat([existing_df, new_df], axis=0)
        fixed_columns = ['timestamp', 'model_id', 'version_id', 'train_date']
        columns_order = fixed_columns + [col for col in concatenate_df.columns if col not in fixed_columns]
        concatenate_df = concatenate_df[columns_order]
        output_dataset.write_with_schema(concatenate_df)
    else:
        logger.info("Schema not compatible, overwriting the metric table.")
        fixed_columns = ['timestamp', 'model_id', 'version']
        columns_order = fixed_columns + [col for col in new_df.columns if col not in fixed_columns]
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