import pandas as pd

def detect_univariate_statistical(
        df,
        sensitivity_score,
        max_fractional_anomalies
):
    df_out = df.assign(is_anomaly=False, anomaly_score=0.0)
    return (df_out, [0,0,0], "No ensemble chosen.")