import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from dataiku.customrecipe import *

from dku_data_drift.drift_analyzer import DriftAnalyzer
from dku_data_drift.model_accessor import ModelAccessor
from model_metadata import get_model_handler

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Model Drift Recipe | %(levelname)s - %(message)s')

# Retrieve array of dataset names from 'input' role, then create datasets
input_names = get_input_names_for_role('original')
original_df = dataiku.Dataset(input_names[0]).get_dataframe()

input_names = get_input_names_for_role('new')
new_df = dataiku.Dataset(input_names[0]).get_dataframe()

output_names = get_output_names_for_role('main_output')
output_datasets = [dataiku.Dataset(name) for name in output_names]
output_dataset =  output_datasets[0]

drifter = DriftAnalyzer()
drifter.fit(new_df=new_df, model_accessor=None, original_df=original_df, target=None)
drift_score = drifter.get_drift_score()

output = {'drift_score': [drift_score]}

output_dataset.write_with_schema(pd.DataFrame(output))
