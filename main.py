
from typing import Callable, Any
import pandas as pd
import flask
import joblib
# from sklearn.pipeline import Pipeline 
from statsmodels.tsa.vector_ar.var_model import VAR

import warnings
warnings.simplefilter(action='ignore')


def init_predict_handler(tag: str = "") -> Callable[[flask.Request], Any]:

    model: VAR = joblib.load(f"artifacts/model{tag}.joblib")  

    def handler() -> Any:
        # request_json = request.get_json()
        model_fit = model.fit() 
        yhat = model_fit.forecast(model_fit.y, steps=1)
        
        return flask.jsonify(dict(forecast=yhat.tolist())) 

    return handler


predict_handler = init_predict_handler()
