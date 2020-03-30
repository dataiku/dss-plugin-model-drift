import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from dataiku.customrecipe import *

from dku_data_drift.drift_analyzer import DriftAnalyzer
from dku_data_drift.model_accessor import ModelAccessor
from model_metadata import get_model_handler

import datetime

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Model Drift Recipe | %(levelname)s - %(message)s')

# Retrieve array of dataset names from 'input' role, then create datasets
input_names = get_input_names_for_role('input')
new_df = dataiku.Dataset(input_names[0]).get_dataframe()

input_names = get_input_names_for_role('model')
model = dataiku.Model(input_names[0])
model_id = model.get_id()

learning_task = get_recipe_config()['learning_task']

output_names = get_output_names_for_role('main_output')
output_datasets = [dataiku.Dataset(name) for name in output_names]
output_dataset =  output_datasets[0]

model_handler = get_model_handler(model)
model_accessor = ModelAccessor(model_handler)
drifter = DriftAnalyzer(prediction_type=learning_task)
target = model_accessor.get_target_variable()
drifter.fit(new_df, model_accessor=model_accessor, original_df=None, target=None)

timestamp_string = datetime.datetime.now().ctime()
drift_score = drifter.get_drift_score()
output = {'model_id': [model_id], 'timestamp': [timestamp_string], 'drift_score': [drift_score]}
output_dataset.write_with_schema(pd.DataFrame(output))
