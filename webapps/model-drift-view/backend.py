import traceback
import logging
from flask import request
import dataiku
from dku_data_drift import DriftAnalyzer, ModelAccessor
from model_metadata import get_model_handler
logger = logging.getLogger(__name__)


@app.route('/list-datasets')
def list_datasets():
    project_key = dataiku.default_project_key()
    client = dataiku.api_client()
    project = client.get_project(project_key)
    dataset_list = [{"name": dataset_dict['name']} for dataset_dict in project.list_datasets()]
    return json.dumps({'dataset_list': dataset_list})

@app.route('/get-drift-metrics')
def get_drift_metrics():
    try:
        model_id = request.args.get('model_id')
        model_version = request.args.get('model_version')
        test_set = request.args.get('test_set')
        new_test_df = dataiku.Dataset(test_set).get_dataframe()

        model = dataiku.Model(model_id)
        model_handler = get_model_handler(model, model_version=model_version)
        model_accessor = ModelAccessor(model_handler)

        drifter = DriftAnalyzer(model_accessor)
        drift_features, drift_clf = drifter.train_drift_model(new_test_df)
        return json.dumps(drifter.compute_drift_metrics(new_test_df, drift_features, drift_clf), allow_nan=False)
    except:
        logger.error(traceback.format_exc())
        return traceback.format_exc(), 500
