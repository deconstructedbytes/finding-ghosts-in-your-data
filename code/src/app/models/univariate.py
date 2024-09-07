import pandas as pd

def detect_univariate_statistical(
        df,
        sensitivity_score,
        max_fractional_anomalies
):
    df_out = df.assign(is_anomaly=False, anomaly_score=0.0)
    return (df_out, [0,0,0], "No ensemble chosen.")





def check_standard_deviation(val:float,
                             mean:float,
                             sd:float,
                             min_num_sd:int):
    """
    Check if a given value is a specified number of 
    standard deviations away from the mean
    """"
    if (abs(val - mean) < (min_num_sd * sd)):
        return abs(val - mean)/(min_num_sd * sd)
    return 1.0

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