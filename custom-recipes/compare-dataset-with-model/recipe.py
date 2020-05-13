import pandas as pd
import json
import datetime
import logging
import dataiku
from dataiku.customrecipe import get_input_names_for_role, get_output_names_for_role, get_recipe_config
from dku_tools import set_column_description, get_train_date
from dku_data_drift.drift_analyzer import DriftAnalyzer
from dku_data_drift.model_accessor import ModelAccessor
from dku_data_drift.dataset_helpers import get_partitioning_columns
from dku_data_drift.model_drift_constants import ModelDriftConstants
from model_metadata import get_model_handler


# init logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Model Drift Recipe | %(levelname)s - %(message)s')

client = dataiku.api_client()
project = client.get_project(dataiku.default_project_key())

# Retrieve input dataset
logger.info("Retrieve the input dataset")
input_names = get_input_names_for_role('input')
new_ds = dataiku.Dataset(input_names[0])
new_df = new_ds.get_dataframe(bool_as_str=True, limit=ModelDriftConstants.MAX_NUM_ROW)

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
active_version = None
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

# Retrieve the output dataset for metrics and score
output_names = get_output_names_for_role('main_output')
output_datasets = [dataiku.Dataset(name) for name in output_names]
output_dataset = output_datasets[0]

# Access the model
model_handler = get_model_handler(model=model, version_id=version_id)
model_accessor = ModelAccessor(model_handler)

# Analyze the drift
drifter = DriftAnalyzer(prediction_type=None)
drifter.fit(new_df, model_accessor=model_accessor)

# Write the drift score and metrics

model_train_date = get_train_date(model_id, version_id)

timestamp = datetime.datetime.now()
new_df = pd.DataFrame({ModelDriftConstants.TIMESTAMP: [timestamp],
                       ModelDriftConstants.MODEL_ID: [model_id],
                       ModelDriftConstants.VERSION_ID: [version_id],
                       ModelDriftConstants.TRAIN_DATE: [model_train_date]})

#fix the column order
new_df = new_df[[ModelDriftConstants.TIMESTAMP, ModelDriftConstants.MODEL_ID, ModelDriftConstants.VERSION_ID, ModelDriftConstants.TRAIN_DATE]]

column_description_dict = {}
if ModelDriftConstants.DRIFT_SCORE in metric_list:
    drift_score = drifter.get_drift_score()
    new_df[ModelDriftConstants.DRIFT_SCORE] = [drift_score]
    column_description_dict[ModelDriftConstants.DRIFT_SCORE] = ModelDriftConstants.DRIFT_SCORE_DEFINITION

if ModelDriftConstants.FUGACITY in metric_list:
    if drifter.get_prediction_type() == 'CLASSIFICATION':
        fugacity, fugacity_relative_change = drifter.get_classification_fugacity()
        new_df[ModelDriftConstants.FUGACITY] = json.dumps(fugacity)
        new_df[ModelDriftConstants.FUGACITY_RELATIVE_CHANGE] = json.dumps(fugacity_relative_change)
        column_description_dict[ModelDriftConstants.FUGACITY] = ModelDriftConstants.FUGACITY_CLASSIF_DEFINITION
        column_description_dict[ModelDriftConstants.FUGACITY_RELATIVE_CHANGE] = ModelDriftConstants.FUGACITY_RELATIVE_CHANGE_CLASSIF_DEFINITION
    else: # regression
        fugacity, fugacity_relative_change, bin_description = drifter.get_regression_fugacity()
        new_df[ModelDriftConstants.FUGACITY] = json.dumps(fugacity)
        new_df[ModelDriftConstants.FUGACITY_RELATIVE_CHANGE] = json.dumps(fugacity_relative_change)
        proper_bin_description = '\n'.join(['Decile {0}: {1}'.format(bin_index, bin_desc) for bin_index, bin_desc in enumerate(bin_description)])
        column_description_dict[ModelDriftConstants.FUGACITY] = ModelDriftConstants.FUGACITY_REGRESSION_DEFINITION + proper_bin_description
        column_description_dict[ModelDriftConstants.FUGACITY_RELATIVE_CHANGE] = ModelDriftConstants.FUGACITY_RELATIVE_CHANGE_REGRESSION_DEFINITION + proper_bin_description


if ModelDriftConstants.FEATURE_IMPORTANCE in metric_list:

    drift_feature_importance = drifter.get_drift_feature_importance()
    original_feature_importance = drifter.get_original_feature_importance()

    if ModelDriftConstants.RISKIEST_FEATURES in metric_list:
        riskiest_feature = drifter.get_riskiest_features(drift_feature_importance, original_feature_importance)
        new_df[ModelDriftConstants.RISKIEST_FEATURES] = json.dumps(riskiest_feature)
        column_description_dict[ModelDriftConstants.RISKIEST_FEATURES] = ModelDriftConstants.RISKIEST_FEATURES_DEFINITION
    feat_dict = {}
    for feat, feat_info in drift_feature_importance[:ModelDriftConstants.NUMBER_OF_DRIFTED_FEATURES].iterrows():
        feat_dict[feat] = round(feat_info.get('importance'), 2)
    new_df[ModelDriftConstants.MOST_DRIFTED_FEATURES] = [json.dumps(feat_dict)]
    column_description_dict[ModelDriftConstants.MOST_DRIFTED_FEATURES] = ModelDriftConstants.MOST_DRIFTED_FEATURES_DEFINITION

    original_feature_importance = drifter.get_original_feature_importance()
    feat_dict = {}
    for feat, feat_info in original_feature_importance[:ModelDriftConstants.NUMBER_OF_DRIFTED_FEATURES].iterrows():
        feat_dict[feat] = round(feat_info.get('importance'), 2)
    new_df[ModelDriftConstants.MOST_IMPORTANT_FEATURES] = [json.dumps(feat_dict)]
    column_description_dict[ModelDriftConstants.MOST_IMPORTANT_FEATURES] = ModelDriftConstants.MOST_DRIFTED_FEATURES_DEFINITION

elif ModelDriftConstants.RISKIEST_FEATURES in metric_list:
    riskiest_feature = drifter.get_riskiest_features()
    new_df[ModelDriftConstants.RISKIEST_FEATURES] = json.dumps(riskiest_feature)
    column_description_dict[ModelDriftConstants.MOST_DRIFTED_FEATURES] = ModelDriftConstants.MOST_DRIFTED_FEATURES_DEFINITION

output_dataset.write_with_schema(new_df)
set_column_description(output_dataset, column_description_dict)