
import time
import json
from datetime import datetime
from typing import Optional, Any

import fire
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR

import warnings
warnings.simplefilter(action='ignore')


np.random.seed(42)

TIMESTAMP_FMT = "%m-%d-%Y, %H:%M:%S"


def train(
    path: str,
    tag: str = "",
    dump: bool = True,
    **kwargs: Optional[Any],
) -> None:

    start = time.time()

    df = pd.read_csv(path, parse_dates=['Time'], index_col='Time', **kwargs)
    
    # missing value treatment
    df['SiteB'][df['SiteB'] == 0] = df['SiteB'].mean()
    
    # checking stationarity 
    cj_stat = coint_johansen(df,-1,1).eig
    
    # creating the train and validation set
    nobs = 1
    df_train, df_test = df[0:-nobs], df[-nobs:]
    
    # fit the model
    model = VAR(df_train)
    model_fit = model.fit()
    
    # Input data for forecasting
    forecast_input = df_train.values[-nobs:]

    # Forecast
    fc = model_fit.forecast(y=forecast_input, steps=nobs)
    df_forecast = pd.DataFrame(fc, index=df.index[-nobs:], columns=df.columns)
    
    # check rmse
    result = dict()
    for i in df.columns: 
        result[f'{i} RMSE'] = np.sqrt(mean_squared_error(df_forecast[i], df_test[i]))
    # print(f"RMSE: {result}")
    
    end = time.time()

    metrics = dict(
        elapsed = end - start,
        cj_stat = cj_stat.tolist(),
        timestamp = datetime.now().strftime(TIMESTAMP_FMT)
    )

    result.update(metrics)
    
    print(f'Metrics: {result}')
    
    # fit the model on full dataset
    model_full = VAR(df) 
    
    if dump:
        joblib.dump(model_full, f"artifacts/model{tag}.joblib")
        json.dump(result, open(f"artifacts/metrics{tag}.json", "w"))


if __name__ == "__main__": 
    fire.Fire(train)
