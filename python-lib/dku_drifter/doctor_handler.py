# -*- coding: utf-8 -*-
import os
import json
import dataiku
from dataiku.doctor.posttraining.model_information_handler import PredictionModelInformationHandler
import logging

logger = logging.getLogger(__name__)


def get_saved_model_version_id(model_id):
    model = dataiku.Model(model_id)
    model_def = model.get_definition()
    saved_model_version_id = 'S-{0}-{1}-{2}'.format(model_def.get('projectKey'), model_def.get('id'), model_def.get('activeVersion'))
    return saved_model_version_id


def get_model_info_handler(saved_model_version_id, datadir_path=os.environ['DIP_HOME']):
    infos = saved_model_version_id.split("-")

    if len(infos) != 4 or infos[0] != "S":
        raise Exception("Invalid saved model id")

    pkey = infos[1]
    model_id = infos[2]
    version_id = infos[3]

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
