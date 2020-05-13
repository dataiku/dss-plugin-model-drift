from dku_data_drift.model_drift_constants import ModelDriftConstants
import numpy as np

def process(dataset, partition_id):
    df = dataset.get_dataframe()
    most_recent_drift_score = df[df[ModelDriftConstants.TIMESTAMP] == np.max(df['timestamp'])]['drift_score'].values[0]
    metric_values = {ModelDriftConstants.DRIFT_SCORE: most_recent_drift_score}
    return metric_values
