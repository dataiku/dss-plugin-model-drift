# -*- coding: utf-8 -*-
import os
import sys
import json
from dataiku.doctor.posttraining.model_information_handler import PredictionModelInformationHandler

def get_model_handler(model, version_id=None):
    saved_model_version_id = _get_saved_model_version_id(model, version_id)
    return _get_model_info_handler(saved_model_version_id)


def _get_model_info_handler(saved_model_version_id):
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
        
    try:
        return PredictionModelInformationHandler(split_desc, core_params, version_folder, version_folder)
    except Exception as e:
        from future.utils import raise_
        if "ordinal not in range(128)" in str(e):
            raise_(Exception, "The plugin is using a python3 code-env, cannot load a python2 model.", sys.exc_info()[2])
        elif str(e) == "non-string names in Numpy dtype unpickling":
            raise_(Exception, "The plugin is using a python2 code-env, cannot load a python3 model.", sys.exc_info()[2])
        else:
            raise_(Exception, "Fail to load saved model: {}".format(e), sys.exc_info()[2])


def _get_saved_model_version_id(model, version_id=None):
    model_def = model.get_definition()
    if version_id is None:
        version_id = model_def.get('activeVersion')
    saved_model_version_id = 'S-{0}-{1}-{2}'.format(model_def.get('projectKey'), model_def.get('id'), version_id)
    return saved_model_version_id
