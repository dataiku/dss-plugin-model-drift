from dku_data_drift.model_drift_constants import ModelDriftConstants
import numpy as np

def process(dataset, partition_id):
    df = dataset.get_dataframe()
    if len(df) == 0:
        return 'No data'
    if ModelDriftConstants.DRIFT_SCORE in df and ModelDriftConstants.TIMESTAMP in df:
        most_recent_drift_score = df[df[ModelDriftConstants.TIMESTAMP] == np.max(df[ModelDriftConstants.TIMESTAMP])][ModelDriftConstants.DRIFT_SCORE].values[0]
        metric_values = {ModelDriftConstants.DRIFT_SCORE: most_recent_drift_score}
        return metric_values
    else:
        return 'No drift score'
