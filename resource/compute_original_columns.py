"""
Allow dynamic select of the model id in the model recipe.
"""
import dataiku
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Model Drift Recipe | %(levelname)s - %(message)s')


def do(payload, config, plugin_config, inputs):
    """
    DSS built-in interface for param loading in the form.
    Retrieve the columns of the original dataset in the drift analysis for multiselect dropdown
    :param payload:
    :param config:
    :param plugin_config:
    :param inputs:
    :return:
    """
    original_ds = None
    for input_role in inputs:
        if input_role["role"] == "original" and input_role["type"] == "DATASET":
            original_ds_name = input_role["fullName"].split('.')[1]
            original_ds = dataiku.Dataset(original_ds_name)
            break
    if original_ds is None:
        raise ValueError
    choice_list = []
    columns = [column['name'] for column in original_ds.read_schema()]
    for column in columns:
        choice_list.append({'value': column, 'label': column})
    return {"choices": choice_list}