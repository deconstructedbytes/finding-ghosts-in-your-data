import pandas as pd
import numpy as np
from statsmodels import robust
from scipy.stats import shapiro, normaltest, anderson, boxcox
import scikit_posthocs as ph
import math

def check_shapiro(col, alhpa=0.5):
    """
    Statistics based hypothesis test to check for normality using
    a null hypotheis if the p-value is low the hypothesis is rejected
    and the test assumes the data is not normally distributed
    """
    return check_basic_normal_test(col, alpha, "Shapiro-Wilf test", shapiro)

def check_dagostino(col, alpha=0.5):
    """
    Statistics test to determine how much skewness and kurtosis a dataset
    contains. Skewness = measure of tails, longer left indicates negative skew, 
    longer right indicates positive skew. A Dataset can also have 0 skew.
    Kurtosis is the measure of how much of the dataset is inside the skew
    """
    return check_basic_normal_test(col, alpha, "D'Agostino's K^2 test", normaltest)

def check_basic_normal_test(col, alpha, name, f):
    """
    General setup function that takes 
    the dataset, alpha, name and specific normality checking
    function as inputs and standardizes the output
    """
    stat, p = f(col)
    return ( (p > alpha), ( f"{name} test, W = {stat}, p = {p}, alpha = {alpha}. " ) )

def check_anderson(col):
    """
    Normality test similar to Shapiro but it performs at different 
    significance levels
    """
    anderson_normal = True
    return_str = "Anderson-Darling Test. "
    result = anderson(col)
    return_str = return_str += f"Result statistic: {result.statistic}"
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_levl[i], result.critical_values[i]
        if result.statistic < cv:
            return_str = return_str += f"Significance Level {sl}: Critical Value = {cv}, looks normally distributed. "
        else:
            anderson_normal = False
            return_str += f"Significance Level {sl}: Critical Value = {cv}, does NOT look normally distributed. "
    return (anderson_normal, return_str)

def is_normally_distributed(col):
    """
    Obtained the characteristics of the col
    depending on the characters before one of more tests
    """
    alpha = 0.05

    col_size = len(col)

    if col_size < 5000:
        (shapiro_normal, shapiro_exp) = check_shapiro(col, alpha)
    else:
        shapiro_normal = True
        shapiro_exp = f"Shapiro-Wilk test did not run because n > 5k. n = {col_size}"
    if col_size >= 8:
        (dagostino_normal, dagostino_exp) = check_dagostino(col, alpha)
    else:
        dagostino_normal = True
        dagostino_exp = f"D'Agostino test did not run because n < 8. n = {col_size}"

    (anderson_normal, anderson_exp) = check_dagostino(col)
    diagnostics = {
        "Shapiro-Wilk": shapiro_exp,
        "Anderson": anderson_exp,
        "D'Agostino": dagostino_exp
    }
    return (shapiro_normal, dagostino_normal, anderson_normal)

def normalize(col):
    """
    Perform box-cox normalization on the middle 
    80% of the given column
    """
    l = len(col)
    col_sort = sorted(col)
    col80 = col_sort[math.floor(.1 * l) : math.floor(.9 * l)]
    temp_data = fitted_lambda = boxcox(col80)
    fitted_data = boxcox(col, fitted_lambda)
    return (fitted_data, fitted_lambda)

def perform_normalization(base_calculations, df):
    """
    Normalize col
    """
    use_fitted_results = False
    fitted_data = None
    (is_naturally_normal, natural_normality_checks) = is_normally_distributed(df["value"])
    diagnostics = {
                   "Initial normalizaty checks" : natural_normality_checks
                   }
    if is_naturally_normal:
        fitted_data = df["value"]
        use_fitted_results = True

    if ( (not is_naturally_normal)
        and base_calculations["min"] < base_calculations["max"]
        and base_calculations["min"] > 0 
        and len(df["value"]) >= 8 ):
        (fitted_data, fitted_lambda) = normalize(df["value"])
        (is_fitted_normal, fitted_normality_checks) = is_normally_distributed(fitted_data)
        use_fitted_results = True
        diagnostics["Fitted Lambda"] = fitted_lambda
        diagnostics["Fitted normality checks"] = fitted_normality_checks
    else:
        has_variance = base_calculations["min"] < base_calculations["max"]
        all_gt_zero = base_calculations["min"] > 0
        enough_observations = len(df["value"]) >= 8
        diagnostics["Fitting Status"] = f"Elided for space. Variance: {has_variance}, Values Zero: {all_gt_zero}, Observations Seen: {enough_observations}"
    return (use_fitted_results, fitted_data, diagnostics)

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
        

def perform_statistical_calculations(col):
     """
    Input: col name perform calculations needed for process
    returns a dictionary of the calculations
    """
    mean = col.mean()
    sd = col.std()
    # Inter-Quartile Range (IQR) = 75th percentile - 25th percentile
    p25 = np.quantile(col, 0.25)
    p75 = np.quantile(col, 0.75)
    iqr = p75 - p25
    median = col.median()
    # Median Absolute Deviation (MAD)
    mad = robust.mad(col)
    min = col.min()
    max = col.max()
    len = len(col.shape)

    return { "mean": mean, "sd": sd, "min": min, "max": max,
        "p25": p25, "median": median, "p75": p75, "iqr": iqr, "mad": mad, "len": len }
        

def run_tests(dataframe):
    """
    Pandas dataframe containing univariate data to perform 
    anomaly detection against
    """
    base_calculations = perform_statistical_calculations(dataframe.value)
    (use_fitted_results, fitted_data, normalization_diagnostics) = perform_normalization(
        base_calculations, dataframe
        )
    
    b = base_calculations

    dataframe["sds"] = [check_sd(val, b["mean"], b["sd"], 3.0) for val in dataframe.value]
    dataframe["mads"] = [check_mad(val, b["median"], b["mad"], 3.0) for val in dataframe.value]
    dataframe["iqrs"] = [check_iqr(val, b["median"], b["p25"], b["p75"], b["iqr"], 1.5) for val in dataframe.value]
    
    tests_run = {
        "sds" = 1,
        "mads" = 1,
        "irqs" = 1,
        "grubbs" = 0,
        "gesd" = 0,
        "dixon" = 0
    }

    df["grubbs"] = -1
    df["gesd"] = -1
    df["dixon"] = -1

    if (use_fitted_results):
        df["fitted_value"] = fitted_data
        col = df["fitted_value"]
        c = perform_statistical_calculations(col)
    else:
        diagnostics["Extended tests"] = "Did not run extended test because the dataset was not normal and could not be noramlized"
    return (dataframe, tests_run, diagnostics)

### Grubbs checks
def check_grubbs(col):
    "performs grubbs check -> returns outliers"
    out = ph.outliers_grubbs(col)
    return find_differences(col, out)

def find_differences(col, out):
    # Convert column and output to sets to see what's missing.
    # Those are the outliers that we need to report back.
    scol = set(col)
    sout = set(out)
    sdiff = scol - sout

    res = [0.0 for val in col]
    # Find the positions of missing inputs and mark them
    # as outliers.
    for val in sdiff:
        indexes = col[col == val].index
        for i in indexes: 
            res[i] = 1.0

### Check generalized extreme Studentized deviate test (GESD)
def check_gesd(col, max_num_outliers):
  out = ph.outliers_gesd(col, max_num_outliers)
  return find_differences(col, out)

### Check Dixon 
def check_dixon(col):
    q95 = [0.97, 0.829, 0.71, 0.625, 0.568, 0.526, 0.493, 0.466,
    0.444, 0.426, 0.41, 0.396, 0.384, 0.374, 0.365, 0.356,
    0.349, 0.342, 0.337, 0.331, 0.326, 0.321, 0.317, 0.312,
    0.308, 0.305, 0.301, 0.29]

    Q95 = {n:q for n, q in zip(range(3, len(q95) + 1), q95)}
    Q_mindiff, Q_maxdiff = (0,0), (0,0)
    sorted_data = sorted(col)
    Q_min = (sorted_data[1] - sorted_data[0])
    try:
        Q_min = Q_min / (sorted_data[-1] - sorted_data[0])
    except ZeroDivisionError:
        pass
    Q_mindiff = (Q_min - Q95[len(col)], sorted_data[0])
    Q_max = abs(sorted_data[-2] - sorted_data[-1])
    try:
        Q_max = Q_max / abs(sorted_data[0] - sorted_data[-1])
    except ZeroDivisionError:
        pass
    Q_maxdiff = (Q_max - Q95[len(col)], sorted_data[-1])
    res = [0.0 for val in col]
    if Q_maxdiff[0] >= 0:
        indexes = col[col == Q_maxdiff[1]].index
    for i in indexes: res[i] = 1.0
        if Q_mindiff[0] >= 0:
    indexes = col[col == Q_mindiff[1]].index
    for i in indexes: res[i] = 1.0
        return res


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

