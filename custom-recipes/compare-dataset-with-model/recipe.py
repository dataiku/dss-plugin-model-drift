import pandas as pd
import datetime
import logging
from dataiku.customrecipe import get_input_names_for_role, get_output_names_for_role, get_recipe_config
from dku_tools import set_column_description, get_train_date, get_input_output, get_params_with_model, build_drift_metric_dataframe
from dku_data_drift.drift_analyzer import DriftAnalyzer
from dku_data_drift.model_accessor import ModelAccessor
from dku_data_drift.dataset_helpers import get_partitioning_columns
from dku_data_drift.model_drift_constants import ModelDriftConstants
from model_metadata import get_model_handler


# init logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Model Drift Recipe | %(levelname)s - %(message)s')

new_dataset, model, output_dataset = get_input_output(has_model_as_second_input=True)
new_df = new_dataset.get_dataframe(bool_as_str=True, limit=ModelDriftConstants.MAX_NUM_ROW)

partition_cols_new_df = get_partitioning_columns(new_dataset)
if partition_cols_new_df:
    new_df = new_df.drop(partition_cols_new_df, axis=1)
if len(new_df.columns) == 0:
    raise ValueError('Without the partition column, dataset is empty.')

version_id, metric_list = get_params_with_model(get_recipe_config(), model)

# Access the model
model_handler = get_model_handler(model=model, version_id=version_id)
model_accessor = ModelAccessor(model_handler)

# Analyze the drift
drifter = DriftAnalyzer(prediction_type=None)
drifter.fit(new_df, model_accessor=model_accessor)

# Write the drift score and metrics
timestamp = datetime.datetime.now()
model_train_date = get_train_date(model.get_id(), version_id)
new_df = pd.DataFrame({ModelDriftConstants.TIMESTAMP: [timestamp],
                       ModelDriftConstants.MODEL_ID: [model.get_id()],
                       ModelDriftConstants.VERSION_ID: [version_id],
                       ModelDriftConstants.TRAIN_DATE: [model_train_date]})
#specify the column order
new_df = new_df[[ModelDriftConstants.TIMESTAMP, ModelDriftConstants.MODEL_ID, ModelDriftConstants.VERSION_ID, ModelDriftConstants.TRAIN_DATE]]
metrics_df, column_description_dict = build_drift_metric_dataframe(drifter, metric_list, new_df, has_model_as_input=True)

output_dataset.write_with_schema(metrics_df)
set_column_description(output_dataset, column_description_dict)