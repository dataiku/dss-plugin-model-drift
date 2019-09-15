# coding: utf-8
import dataiku
import os
import json
from dataiku.doctor.posttraining.model_information_handler import PredictionModelInformationHandler

def get_saved_model_version_id(model, model_version=None):
    if model_version is None:
        model_version = model_def.get('activeVersion')
    model_def = model.get_definition()
    saved_model_version_id = 'S-{0}-{1}-{2}'.format(model_def.get('projectKey'), model_def.get('id'), model_version)
    return saved_model_version_id

def get_model_handler(model, model_version=None):
    my_data_dir = os.environ['DIP_HOME']
    saved_model_version_id = get_saved_model_version_id(model, model_version)
    return get_model_info_handler(saved_model_version_id)

def get_model_info_handler(saved_model_version_id):
    infos = saved_model_version_id.split("-")

    if len(infos) != 4 or infos[0] != "S":
        raise Exception("Invalid saved model id")

    pkey = infos[1]
    model_id = infos[2]
    version_id = infos[3]

    datadir_path = os.environ['DIP_HOME']
    version_folder = os.path.join(datadir_path, "saved_models", pkey, model_id, "versions", version_id)

    # Loading and resolving paths in split_desc
    split_folder = os.path.join(version_folder, "split")
    with open(os.path.join(split_folder, "split.json")) as split_file:
        split_desc = json.load(split_file)

    path_field_names = ["trainPath", "testPath", "fullPath"]
    for field_name in path_field_names:
        if split_desc.get(field_name, None) is not None:
            split_desc[field_name] = os.path.join(split_folder, split_desc[field_name])

    with open(os.path.join(version_folder, "core_params.json")) as core_params_file:
        core_params = json.load(core_params_file)

    return PredictionModelInformationHandler(split_desc, core_params, version_folder, version_folder)