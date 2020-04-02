import dataiku
import pandas as pd, numpy as np
from dataiku.customrecipe import *

from dku_data_drift.drift_analyzer import DriftAnalyzer

import datetime
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Model Drift Recipe | %(levelname)s - %(message)s')

# Retrieve the reference dataset
input_names = get_input_names_for_role('original')
original_df = dataiku.Dataset(input_names[0]).get_dataframe()

# Retrieve the new dataset
input_names = get_input_names_for_role('new')
new_df = dataiku.Dataset(input_names[0]).get_dataframe()

# Retrieve the target variable
target_variable = get_recipe_config().get('target_variable')
if target_variable is None:
    raise ValueError('Target variable must be defined.')

# Retrieve the output dataset where the metrics and drift score will be written
output_names = get_output_names_for_role('main_output')
output_datasets = [dataiku.Dataset(name) for name in output_names]
output_dataset = output_datasets[0]

# Retrieve the learning task
learning_task = get_recipe_config().get('learning_task')
if learning_task is None:
    raise ValueError('Learning task must be defined.')

# Analyse the drift
drifter = DriftAnalyzer(learning_task)
drifter.fit(new_df=new_df, original_df=original_df, target=target_variable)

# Write the drift score and metrics
timestamp = datetime.datetime.now()
drift_score = drifter.get_drift_score()
output = {'timestamp': [timestamp], 'drift_score': [drift_score]}
output_dataset.write_with_schema(pd.DataFrame(output, columns=['timestamp', 'drift_score']))
