import pandas as pd
import numpy as np
from statsmodels import robust

def detect_univariate_statistical(
        dataframe,
        sensitivity_score,
        max_fractional_anomalies
):
    df_out = df.assign(is_anomaly=False, anomaly_score=0.0)
    return (df_out, [0,0,0], "No ensemble chosen.")


## We also need a function to determine the median absolute devation 
#  from the median, since the code is similar to the 
#  check_standard_deviation check we can abstract a more general
#  check distances from "middle" value checks

def check_stat(val:float,
               midpoint:float,
               distance:int,
               n:int):
    """
    Check if a given value is within a given range of a 
    midpoint value and a number of increments. If the value is within 
    this range return a percentage else return 1.0 indicating the value 
    is an statistical outlier
    """
    if (abs(val - midpoint) < (n * distance)):
        return abs(val-midpoint) / (n * distance)
    return 1.0

def check_sd(val:float,
             mean:float,
             sd:float,
             min_num_sd:int):
     """
     Check if a given value is a specified number of 
     standard deviations away from the mean
     """
     return check_stat(val, mean, sd, min_num_sd)

def check_mad(val:float,
              median:float,
              mad:float,
              min_num_mad:int):
    """
    Check if a given value is with the range of
    the median absolute value and a specific length or distance
    If the value is within the range return a percentage, else
    return 1.o indicating it is an outlier
    """
    return check_stat(val, median, mad, min_num_mad)

def check_iqr(val:float,
              median:float,
              p25:float,
              p75:float,
              iqr:float,
              min_iqr_diff:float):
    """
    Check if on which side of the median a value exists
    If below the median checks if the value is min_iqr_diff times below the p25 IQR
    if above checks if the value min_iqr_diff times above the p75.
    if the value passes those checks return 1.0 to suggest the value
    is an outlier
    """
    if val < median:
        if val > p25:
             return 0.0
        elif (p25 - val) < (min_iqr_diff * iqr):
             return abs(p25 - val) / (min_iqr_diff * iqr)
        else:
            return 1.0
    else:
        if val < p75:
            return 0.0
        elif (val - p75) < (min_iqr_diff * iqr):
            return abs(val - p75) / (min_iqr_diff * iqr)
        else:
            return 1.0
        

def run_tests(dataframe):
    """
    Pandas dataframe containing univariate data to perform 
    anomaly detection against
    """
    mean = dataframe.value.mean()
    sd = dataframe.value.std(0)
    p25 = np.quantile(dataframe.value, 0.25)
    p75 = np.quantile(dataframe.value, 0.75)
    iqr = p75 - p25
    median = dataframe.value.median()
    mad = robust.mad(dataframe.value)
    calculations = {
        "mean": mean, "sd": sd, "p25": p25,
        "p75": p75, "iqr": iqr, "median": median,
        "mad":mad
    }
    dataframe["sds"] = [check_sd(val, mean, sd, 3.0) for val in dataframe.value]
    dataframe["mads"] = [check_mad(val, median, mad, 3.0) for val in dataframe.value]
    dataframe["iqrs"] = [check_iqr(val, median, p25, p75, iqr, 1.5) for val in dataframe.value]
    
    return (dataframe, calculations)
    
def score_results(
        dataframe,
        weights
):
    """
    Take a dataframe and dictionary of weights
    """
    return dataframe.assign(anomaly_score=(
        dataframe["sds"] * weights["sds"] + 
        dataframe["iqrs"] * weights["iqrs"] +
        dataframe["mads"] * weights["mads"]
    ))

def determine_outliers(
        dataframe,
        sensitivity_score,
        max_fractional_anomalies
):
    sensitivity_score = (100 -  sensitivity_score) / 100.0
    max_fractional_anomaly_score = np.quantile(dataframe.anomaly_score,
                                           1.0 - max_fractional_anomalies)
    if max_fractional_anomaly_score > sensitivity_score and max_fractional_anomalies < 1.0:
        sensitivity_score = max_fractional_anomaly_score
        
    return dataframe.assign(
        is_anomaly=(dataframe.anomaly_score > sensitivity_score)
        )
    


def detect_univariate_statistical(
        dataframe,
        sensitivity_score,
        max_fractional_anomalies
):
    weights = {
        "sds": 0.25,
        "iqrs": 0.35,
        "mads": 0.45
    }
    #print(dataframe)
    if (dataframe.value.count() < 3):
        return (dataframe.assign(is_anomaly=False, anomaly_score=0.0), weights, "Must have minimum of 3 items for anomaly detection")
    elif (max_fractional_anomalies <= 0.0 or max_fractional_anomalies > 1.0):
        return (dataframe.assign(is_anomaly=False, anomaly_score=0.0), weights, "Must have valid max fraction of anomalies, 0 < x <= 1.0")
    elif (sensitivity_score <= 0 or sensitivity_score > 100):
        return (dataframe.assign(is_anomaly=False, anomaly_score=0.0), weights, "Must have valid sensitivity score, 0 < x <= 100.0")
    else:
        df_test, calculations = run_tests(dataframe)
        df_scored = score_results(df_test, weights)
        df_out = determine_outliers(df_scored, sensitivity_score, max_fractional_anomalies)
        return  (df_out, weights, {"message" : "Ensemble of [mean +/- 3*SD, median +/- 3*MAD, median +/- 1.5*IQR],",
                                "calculations": calculations}) 

