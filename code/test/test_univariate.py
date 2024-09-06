from src.app.models.univariate import *
import pandas as pd
import pytest


@pytest.mark.parametrize("df_input", [
    [1,2,3,4,5,6,7,8,9,10], 
    [1], 
    [1,2,3,4,5.6,7.8,9,10], 
    []
])
def test_detect_univariate_statisticals_returns_correct_number_of_rows(df_input):
    # Arrange
    df = pd.DataFrame(df_input, columns=["value"])
    sensivity_score = 75
    max_fraction_anomalies = 0.20
    # Act
    df_out, weights, details = detect_univariate_statistical(df, sensivity_score, max_fraction_anomalies)
    # Assert
    assert(len(df) == len(df_out))