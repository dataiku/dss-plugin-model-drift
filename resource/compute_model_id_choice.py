"""
Allow dynamic select of the model id in the model recipe.
"""
import dataiku


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
    available_model_ids = [model_desc['id'] for model_desc in available_model_version]
    choices = [ {"value": id_, "label": id_} for id_ in available_model_ids]
    return {"choices": choices}