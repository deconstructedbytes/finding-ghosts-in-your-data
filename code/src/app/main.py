from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import json
import datetime


app = FastAPI()

@app.get("/")
def doc():
    return {
        "message" : "Welcome to the anomaly detector service",
        "documentation" : "If you want to see the OpenAPI specifications navigate to the /redoc/ path on this server"
    }
class Univariate_Statistical_Input(BaseModel):
    key: str
    value: float


@app.post("/detect/univariate")
def post_univariate(
    input_data: List[Univariate_Statistical_Input],
    sensitivitiy_score: float = 50,
    max_fractional_anomalies: float = 1.0,
    debug: bool = False,
    <other inputs>
    ):
    df = pd.DataFrame(i.__dict__ for i in input_data)
    (df, weights, details) = univariate.detect_univariate_statistical(
        df, sensitivitiy_score, max_fractional_anomalies
        )
    results = {"anomalies" : json.loads(df.to_json(orient='records')) }
    if (debug):
        #TODO: add debug data
        results.update( {"debug_msg": "This is a logging message" })
        results.update( {"debug_weights": weights }) 
        results.update( {"debug_details": details })

    return results
