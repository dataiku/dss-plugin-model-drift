import dataiku
import pandas as pd
import json
from dataiku.customrecipe import *
from dku_tools import set_column_description, get_train_date
from dku_data_drift.drift_analyzer import DriftAnalyzer
from dku_data_drift.model_accessor import ModelAccessor
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
new_df = new_ds.get_dataframe(bool_as_str=True, limit=MAX_NUM_ROW)

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

# Retrieve the version id of the model (dynamic dropdown selection)
use_active_version = get_recipe_config().get('use_active_version')
if use_active_version:
    for version in model.list_versions():
        active_version = version.get('active') is True
        if active_version:
            version_id = version.get('versionId')
            break
else:
    version_id = get_recipe_config().get('version_id')
    if version_id is None:
        raise ValueError('Please choose a model version.')

metric_list = get_recipe_config().get('metric_list')
if len(metric_list) == 0 or metric_list is None:
    raise ValueError('Please choose at least one metric.')
logger.info('Chosen metrics: ', metric_list)

output_format = get_recipe_config().get('output_format', 'multiple_columns')

# Retrieve the output dataset for metrics and score
output_names = get_output_names_for_role('main_output')
output_datasets = [dataiku.Dataset(name) for name in output_names]
output_dataset = output_datasets[0]

# Access the model
model_handler = get_model_handler(model=model, version_id=version_id)
model_accessor = ModelAccessor(model_handler)

# Analyze the drift
drifter = DriftAnalyzer(prediction_type=None)
target = model_accessor.get_target_variable()
drifter.fit(new_df, model_accessor=model_accessor)

# Write the drift score and metrics

model_train_date = get_train_date(model_id, version_id)

timestamp = datetime.datetime.now()
new_df = pd.DataFrame({'timestamp': [timestamp],
                       'model_id': [model_id],
                       'version_id': [version_id],
                       'train_date': [model_train_date]})

#fix the column order
new_df = new_df[['timestamp', 'model_id', 'version_id', 'train_date']]

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
    drift_feature_importance = drifter.get_drift_feature_importance()
    original_feature_importance = drifter.get_original_feature_importance()
    riskiest_feature = drifter.get_riskiest_features(drift_feature_importance, original_feature_importance)
    new_df['riskiest_feature'] = json.dumps(riskiest_feature)
    column_description_dict['riskiest_feature'] = 'If the drift score is low, we recommend you to check those features'

    if output_format == 'multiple_columns':
        feat_dict = {}
        for feat, feat_info in drift_feature_importance[:10].iterrows():
            feat_dict[feat] = round(feat_info.get('importance'), 2)
        new_df['drift_feature_importance'] = [json.dumps(feat_dict)]
        column_description_dict['drift_feature_importance'] = 'Features that have been drifted the most, with their % of importance'

        original_feature_importance = drifter.get_original_feature_importance()
        feat_dict = {}
        for feat, feat_info in original_feature_importance[:10].iterrows():
            feat_dict[feat] = round(feat_info.get('importance'), 2)
        new_df['original_feature_importance'] = [json.dumps(feat_dict)]
        column_description_dict['original_feature_importance'] = 'Most important features in the deployed model, with their % of importance'
    else:

        drift_feature_importance = drifter.get_drift_feature_importance()
        tmp_dict_drift = {}
        for feat, feat_info in drift_feature_importance[:10].iterrows():
            tmp_dict_drift[feat] = round(feat_info.get('importance'), 2)


        tmp_dict_original = {}
        for feat, feat_info in original_feature_importance[:10].iterrows():
            tmp_dict_original[feat] = round(feat_info.get('importance'), 2)

        feat_dict = {}
        feat_dict['drift_feature_importance'] = tmp_dict_drift
        feat_dict['original_feature_importance'] = tmp_dict_original

        new_df['feature_importance'] = json.dumps(feat_dict)
        column_description_dict['feature_importance'] = 'drift_feature_importance: List of features that have been drifted the most, with their % of importance. original_feature_importance: List of the most important features in the deployed model, with their % of importance'


output_dataset.write_with_schema(new_df)
set_column_description(output_dataset, column_description_dict)