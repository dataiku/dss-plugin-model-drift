import pandas as pd
import json
import datetime
import logging
from dataiku.customrecipe import get_input_names_for_role, get_output_names_for_role, get_recipe_config
from dku_tools import set_column_description, get_train_date, get_input_output, get_params_with_model
from dku_data_drift.drift_analyzer import DriftAnalyzer
from dku_data_drift.model_accessor import ModelAccessor
from dku_data_drift.dataset_helpers import get_partitioning_columns
from dku_data_drift.model_drift_constants import ModelDriftConstants
from model_metadata import get_model_handler


# init logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Model Drift Recipe | %(levelname)s - %(message)s')

new_dataset, model, output_dataset = get_input_output(has_model_as_second_input=True)
new_df = new_dataset.get_dataframe(bool_as_str=True, limit=ModelDriftConstants.MAX_NUM_ROW)

partition_cols_new_df = get_partitioning_columns(new_dataset)
if partition_cols_new_df:
    new_df = new_df.drop(partition_cols_new_df, axis=1)
if len(new_df.columns)==0:
    raise ValueError('Without the partition column, dataset is empty.')

version_id, metric_list = get_params_with_model(get_recipe_config(), model)

# Access the model
model_handler = get_model_handler(model=model, version_id=version_id)
model_accessor = ModelAccessor(model_handler)

# Analyze the drift
drifter = DriftAnalyzer(prediction_type=None)
drifter.fit(new_df, model_accessor=model_accessor)

# Write the drift score and metrics
timestamp = datetime.datetime.now()
model_train_date = get_train_date(model.get_id(), version_id)
new_df = pd.DataFrame({ModelDriftConstants.TIMESTAMP: [timestamp],
                       ModelDriftConstants.MODEL_ID: [model.get_id()],
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
    if drifter.get_prediction_type() == ModelDriftConstants.CLASSIFICATION_TYPE:
        fugacity, fugacity_relative_change = drifter.get_classification_fugacity()
        new_df[ModelDriftConstants.FUGACITY] = json.dumps(fugacity)
        new_df[ModelDriftConstants.FUGACITY_RELATIVE_CHANGE] = json.dumps(fugacity_relative_change)
        column_description_dict[ModelDriftConstants.FUGACITY] = ModelDriftConstants.FUGACITY_CLASSIF_DEFINITION
        column_description_dict[ModelDriftConstants.FUGACITY_RELATIVE_CHANGE] = ModelDriftConstants.FUGACITY_RELATIVE_CHANGE_CLASSIF_DEFINITION
    elif drifter.get_prediction_type() == ModelDriftConstants.REGRRSSION_TYPE:
        fugacity, fugacity_relative_change, bin_description = drifter.get_regression_fugacity()
        new_df[ModelDriftConstants.FUGACITY] = json.dumps(fugacity)
        new_df[ModelDriftConstants.FUGACITY_RELATIVE_CHANGE] = json.dumps(fugacity_relative_change)
        proper_bin_description = '\n'.join(['Decile {0}: {1}'.format(bin_index, bin_desc) for bin_index, bin_desc in enumerate(bin_description)])
        column_description_dict[ModelDriftConstants.FUGACITY] = ModelDriftConstants.FUGACITY_REGRESSION_DEFINITION + proper_bin_description
        column_description_dict[ModelDriftConstants.FUGACITY_RELATIVE_CHANGE] = ModelDriftConstants.FUGACITY_RELATIVE_CHANGE_REGRESSION_DEFINITION + proper_bin_description
    else:
        raise ValueError('Unsupported prediction type: {0}'.format(drifter.get_prediction_type()))


if ModelDriftConstants.FEATURE_IMPORTANCE in metric_list:

    drift_feature_importance = drifter.get_drift_feature_importance()
    original_feature_importance = drifter.get_original_feature_importance()

    if ModelDriftConstants.RISKIEST_FEATURES in metric_list:
        riskiest_feature = drifter.get_riskiest_features(drift_feature_importance, original_feature_importance)
        new_df[ModelDriftConstants.RISKIEST_FEATURES] = json.dumps(riskiest_feature)
        column_description_dict[ModelDriftConstants.RISKIEST_FEATURES] = ModelDriftConstants.RISKIEST_FEATURES_DEFINITION
    feat_dict = {}
    for feat, feat_info in drift_feature_importance[:ModelDriftConstants.NUMBER_OF_DRIFTED_FEATURES].iterrows():
        feat_dict[feat] = round(feat_info.get(ModelDriftConstants.IMPORTANCE), 2)
    new_df[ModelDriftConstants.MOST_DRIFTED_FEATURES] = [json.dumps(feat_dict)]
    column_description_dict[ModelDriftConstants.MOST_DRIFTED_FEATURES] = ModelDriftConstants.MOST_DRIFTED_FEATURES_DEFINITION

    original_feature_importance = drifter.get_original_feature_importance()
    feat_dict = {}
    for feat, feat_info in original_feature_importance[:ModelDriftConstants.NUMBER_OF_DRIFTED_FEATURES].iterrows():
        feat_dict[feat] = round(feat_info.get(ModelDriftConstants.IMPORTANCE), 2)
    new_df[ModelDriftConstants.MOST_IMPORTANT_FEATURES] = [json.dumps(feat_dict)]
    column_description_dict[ModelDriftConstants.MOST_IMPORTANT_FEATURES] = ModelDriftConstants.MOST_DRIFTED_FEATURES_DEFINITION

elif ModelDriftConstants.RISKIEST_FEATURES in metric_list:
    riskiest_feature = drifter.get_riskiest_features()
    new_df[ModelDriftConstants.RISKIEST_FEATURES] = json.dumps(riskiest_feature)
    column_description_dict[ModelDriftConstants.MOST_DRIFTED_FEATURES] = ModelDriftConstants.MOST_DRIFTED_FEATURES_DEFINITION
else:
    pass

output_dataset.write_with_schema(new_df)
set_column_description(output_dataset, column_description_dict)