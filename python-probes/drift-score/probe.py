import numpy as np

def process(dataset, partition_id):
    df = dataset.get_dataframe()
    most_recent_drift_score = df[df['timestamp'] == np.max(df['timestamp'])]['drift_score'].values[0]
    metric_values = {'Drift score': most_recent_drift_score}
    return metric_values
