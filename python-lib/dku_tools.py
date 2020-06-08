# -*- coding: utf-8 -*-
import datetime
import json
import dataiku
from dataiku.customrecipe import get_input_names_for_role, get_output_names_for_role
from dku_data_drift.model_drift_constants import ModelDriftConstants


def process_timestamp(timestamp):
    """
    Convert the timestamp to str date
    :param timestamp:
    :return:
    """
    return str(datetime.datetime.fromtimestamp(timestamp / 1000))


def get_train_date(model_version, version_id):
    m = dataiku.Model(model_version, ignore_flow=True)
    for v in m.list_versions():
        if v.get('versionId') == version_id:
            return process_timestamp((v.get('snippet').get('trainDate')))
    return None


def set_column_description(dataset, column_description_dict):
    dataset_schema = dataset.read_schema()
    for col_info in dataset_schema:
        col_name = col_info.get('name')
        col_info['comment'] = column_description_dict.get(col_name)
    dataset.write_schema(dataset_schema)


def get_input_output(has_model_as_second_input=False):

    if len(get_input_names_for_role('new')) == 0:
        raise ValueError('No new dataset.')
    if len(get_output_names_for_role('output_dataset')) == 0:
        raise ValueError('No output dataset.')

    new_dataset_name = get_input_names_for_role('new')[0]
    new_dataset = dataiku.Dataset(new_dataset_name)

    output_dataset_name = get_output_names_for_role('output_dataset')[0]
    output_dataset = dataiku.Dataset(output_dataset_name)

    if has_model_as_second_input:
        if len(get_input_names_for_role('model')) == 0:
            raise ValueError('No input model.')
        model_name = get_input_names_for_role('model')[0]
        model = dataiku.Model(model_name)
        return (new_dataset, model, output_dataset)
    else:
        if len(get_input_names_for_role('original')) == 0:
            raise ValueError('No original dataset.')

        original_dataset_name = get_input_names_for_role('original')[0]
        original_dataset = dataiku.Dataset(original_dataset_name)
        return (new_dataset, original_dataset, output_dataset)


def get_params_with_model(recipe_config, model):
    use_active_version = recipe_config.get('use_active_version')
    active_version = None
    if use_active_version:
        for version in model.list_versions():
            active_version = version.get('active') is True
            if active_version:
                version_id = version.get('versionId')
                break
    else:
        version_id = recipe_config.get('version_id')
        if version_id is None:
            raise ValueError('Please choose a model version.')

    metric_list = recipe_config.get('metric_list')
    if len(metric_list) == 0 or metric_list is None:
        raise ValueError('Please choose at least one metric.')
    return version_id, metric_list


def get_params_without_model(recipe_config):
    metric_list = recipe_config.get('metric_list_without_prediction')
    if len(metric_list) == 0 or metric_list is None:
        raise ValueError('Please choose at least one metric.')

    # Handle columns to remove
    columns_to_remove = recipe_config.get('columns_to_remove')
    return columns_to_remove, metric_list


def build_drift_metric_dataframe(drifter, metric_list, based_df, has_model_as_input):

    new_df = based_df.copy()
    column_description_dict = {}

    if ModelDriftConstants.DRIFT_SCORE in metric_list:
        # new_df_with_drift_score, column_description_dict = extract_drift_score(drifter, new_df, column_description_dict)
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
        feat_dict = {}
        for feat, feat_info in drift_feature_importance[:ModelDriftConstants.NUMBER_OF_DRIFTED_FEATURES].iterrows():
            feat_dict[feat] = round(feat_info.get(ModelDriftConstants.IMPORTANCE), 2)
        new_df[ModelDriftConstants.MOST_DRIFTED_FEATURES] = [json.dumps(feat_dict)]
        column_description_dict[ModelDriftConstants.MOST_DRIFTED_FEATURES] = ModelDriftConstants.MOST_DRIFTED_FEATURES_DEFINITION

        if has_model_as_input:
            original_feature_importance = drifter.get_original_feature_importance()
            feat_dict = {}
            for feat, feat_info in original_feature_importance[:ModelDriftConstants.NUMBER_OF_DRIFTED_FEATURES].iterrows():
                feat_dict[feat] = round(feat_info.get(ModelDriftConstants.IMPORTANCE), 2)
            new_df[ModelDriftConstants.MOST_IMPORTANT_FEATURES] = [json.dumps(feat_dict)]
            column_description_dict[ModelDriftConstants.MOST_IMPORTANT_FEATURES] = ModelDriftConstants.MOST_DRIFTED_FEATURES_DEFINITION

    if ModelDriftConstants.RISKIEST_FEATURES in metric_list:
        riskiest_feature = drifter.get_riskiest_features()
        new_df[ModelDriftConstants.RISKIEST_FEATURES] = json.dumps(riskiest_feature)
        column_description_dict[ModelDriftConstants.MOST_DRIFTED_FEATURES] = ModelDriftConstants.MOST_DRIFTED_FEATURES_DEFINITION

    return new_df, column_description_dict