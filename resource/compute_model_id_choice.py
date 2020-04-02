"""
Allow dynamic select of the model id in the model recipe.
"""
import dataiku
import datetime


def process_timestamp(timestamp):
    """
    Convert the timestamp to str date
    :param timestamp:
    :return:
    """
    return str(datetime.datetime.fromtimestamp(timestamp / 1000))


def do(payload, config, plugin_config, inputs):
    """
    DSS built-in interface for param loading in the form.
    Retrieve the available versions of a pretrained model in DSS.
    :param payload:
    :param config:
    :param plugin_config:
    :param inputs:
    :return:
    """
    model = None
    for input_ in inputs:
        if input_['role'] == 'model':
            model = str(input_['fullName'])
    if model is None:
        raise Exception("Did not catch the right input model")
    model_id = model.split('.')[-1]
    client = dataiku.api_client()
    project = client.get_project(dataiku.default_project_key())
    saved_model = project.get_saved_model(model_id)
    available_model_version = saved_model.list_versions()
    available_model = [(model_desc['id'],  process_timestamp(model_desc['trainDate'])) for model_desc in available_model_version]
    choices = [ {"value": model_[0], "label": "version {}, deployed at {}".format(*model_)} for model_ in available_model]
    return {"choices": choices}