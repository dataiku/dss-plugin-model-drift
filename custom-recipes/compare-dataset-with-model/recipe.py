import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from dataiku.customrecipe import *

from dku_data_drift.drift_analyzer import DriftAnalyzer
from dku_data_drift.model_accessor import ModelAccessor
from model_metadata import get_model_handler

import datetime
import logging

# init logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Model Drift Recipe | %(levelname)s - %(message)s')

client = dataiku.api_client()
project = client.get_project(dataiku.default_project_key())

# Retrieve input dataset
logger.info("Retrieve the input dataset")
input_names = get_input_names_for_role('input')
new_df = dataiku.Dataset(input_names[0]).get_dataframe(limit=100000)

# Retrieve input model
logger.info("Retrieve the input model")
input_names = get_input_names_for_role('model')
model = dataiku.Model(input_names[0])
model_id = model.get_id()
saved_model = project.get_saved_model(model_id)
available_model_version = saved_model.list_versions()
logger.info("The input model has the following version ids: {}".format(available_model_version))

# Retrieve the version id of the model (dynamic dropdown selection)
version_id = get_recipe_config().get('version_id')
if version_id is None:
    raise ValueError('Version id must be defined.')

# Retrieve the output dataset for metrics and score
output_names = get_output_names_for_role('main_output')
output_datasets = [dataiku.Dataset(name) for name in output_names]
output_dataset =  output_datasets[0]

# Access the model
model_handler = get_model_handler(model=model, version_id=version_id)
model_accessor = ModelAccessor(model_handler)

# Analyze the drift
drifter = DriftAnalyzer(prediction_type=None)
target = model_accessor.get_target_variable()
drifter.fit(new_df, model_accessor=model_accessor)

# Write the metrics and information about the drift in an output dataset
timestamp = datetime.datetime.now()
drift_score = drifter.get_drift_score()
output = {'timestamp': [timestamp], 'model_id': [model_id], 'model_version': [version_id], 'drift_score': [drift_score]}
output_dataset.write_with_schema(pd.DataFrame(output, columns=['timestamp', 'model_id', 'model_version', 'drift_score']))
