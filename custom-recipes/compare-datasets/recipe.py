import pandas as pd
import datetime
import logging
from dataiku.customrecipe import get_input_names_for_role, get_output_names_for_role, get_recipe_config
from dku_data_drift.drift_analyzer import DriftAnalyzer
from dku_data_drift.dataset_helpers import get_partitioning_columns
from dku_data_drift.model_drift_constants import ModelDriftConstants
from dku_tools import set_column_description, get_input_output, get_params_without_model, build_drift_metric_dataframe

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Model Drift Recipe | %(levelname)s - %(message)s')

new_dataset, original_dataset, output_dataset = get_input_output()
original_df = original_dataset.get_dataframe(bool_as_str=True, limit=ModelDriftConstants.MAX_NUM_ROW)
new_df = new_dataset.get_dataframe(bool_as_str=True, limit=ModelDriftConstants.MAX_NUM_ROW)

columns_to_remove, metric_list = get_params_without_model(get_recipe_config())

if len(columns_to_remove) != 0:
    to_remove_in_original = set(original_df.columns).intersection(set(columns_to_remove))
    if to_remove_in_original:
        original_df = original_df.drop(list(to_remove_in_original), axis=1)
    to_remove_in_new = set(new_df.columns).intersection(set(columns_to_remove))
    if to_remove_in_new:
        new_df = new_df.drop(list(to_remove_in_new), axis=1)

# Handle partitioning
partition_cols_new_df = get_partitioning_columns(new_dataset)
partition_cols_original_df = get_partitioning_columns(original_dataset)
if partition_cols_original_df:
    original_df = original_df.drop(partition_cols_original_df, axis=1)
if partition_cols_new_df:
    new_df = new_df.drop(partition_cols_new_df, axis=1)
if len(new_df.columns) == 0 or len(original_df.columns) == 0:
    raise ValueError('Without the partition column, at least one of the datasets is empty.')

# Analyse the drift
drifter = DriftAnalyzer()
drifter.fit(new_df=new_df, original_df=original_df)

# Write the drift score and metrics
timestamp = datetime.datetime.now()
new_df = pd.DataFrame({ModelDriftConstants.TIMESTAMP: [timestamp]})
metrics_df, column_description_dict = build_drift_metric_dataframe(drifter, metric_list, new_df, has_model_as_input=False)

output_dataset.write_with_schema(metrics_df)
set_column_description(output_dataset, column_description_dict)

