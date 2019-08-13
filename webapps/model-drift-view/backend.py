import dataiku
from flask import request

from dku_drifter import DriftAnalyzer, ModelAccessor
from commons import get_model_handler

@app.route('/get_drift_metrics')
def get_drift_metrics():
    model_id = request.args.get('model_id')
    test_set = request.args.get('test_set')
    new_test_df = dataiku.Dataset(test_set).get_dataframe()

    model = dataiku.Model(model_id)
    model_handler = get_model_handler(model)
    model_accessor = ModelAccessor(model_handler)

    drifter = DriftAnalyzer(model_accessor)
    drift_features, drift_clf = drifter.train_drift_model(new_test_df)
    drift_metrics = drifter.generate_drift_metrics(new_test_df, drift_features, drift_clf)   
    return drift_metrics