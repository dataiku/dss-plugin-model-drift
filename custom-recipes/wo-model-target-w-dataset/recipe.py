import dataiku
import pandas as pd, numpy as np
from dataiku.customrecipe import *

from dku_data_drift.drift_analyzer import DriftAnalyzer

import datetime
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Model Drift Recipe | %(levelname)s - %(message)s')

# Retrieve reference dataset from input
input_names = get_input_names_for_role('original')
original_df = dataiku.Dataset(input_names[0]).get_dataframe()

# Retrieve the new dataset from input
input_names = get_input_names_for_role('new')
new_df = dataiku.Dataset(input_names[0]).get_dataframe()

# Retrieve the output dataset
output_names = get_output_names_for_role('main_output')
output_datasets = [dataiku.Dataset(name) for name in output_names]
output_dataset = output_datasets[0]

# Retrieve the learning task
learning_task = get_recipe_config()['learning_task']

# Analyse the drift
drifter = DriftAnalyzer(learning_task)
drifter.fit(new_df=new_df, model_accessor=None, original_df=original_df, target=None)

# Write metrics and drift score in output dataset
timestamp = datetime.datetime.now()
drift_score = drifter.get_drift_score()
output = {'timestamp': [timestamp], 'drift_score': [drift_score]}
try:
    existing_df = output_dataset.get_dataframe()
    logger.info("Got existing dataframe")
except:
    logger.info("No existing dataframe")
    existing_df = None
output_df = pd.DataFrame(output)
if existing_df is None:
    output_dataset.write_with_schema(output_df)
else:
    output_dataset.write_with_schema(pd.concat([existing_df, output_df], axis=0))
logger.info("Recipe processing has ended.")
