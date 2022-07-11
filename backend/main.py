from fastapi import FastAPI
from fastapi.responses import JSONResponse
# from pydantix import BaseModel
import pandas as pd

import uvicorn
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from forecast.model import *
from forecast.preprocess import forecast_preprocess

app = FastAPI()
yscaler = pickle.load(open('/Users/kim_yoonhye/Desktop/TS-컨퍼/github/WebServing/forecast/scaler.pkl', 'rb'))


def load_forecast_model():
    model = NBeatsNet(stack_types=(NBeatsNet.TREND_BLOCK, NBeatsNet.SEASONALITY_BLOCK),
                        forecast_length=48,
                        backcast_length=144,
                        hidden_layer_units=128,
                        )
    model.load_state_dict(torch.load('/Users/kim_yoonhye/Desktop/TS-컨퍼/github/WebServing/forecast/model.pth'))
    return model


@app.get("/forecast")
def get_forecast(data : str):
    df = pd.read_json(data)
    forecast_df = forecast_preprocess(df, yscaler)
    sample_size = 50
    prediction_size = 48

    fc_model = load_forecast_model()
    y_pred = np.zeros((sample_size, prediction_size))
    trend = np.zeros((sample_size, prediction_size))
    seasonality = np.zeros((sample_size, prediction_size))

    for i in range(sample_size):
        backcast, pred, trend_forecast, seasonal_forecast = fc_model(forecast_df)
        pred = pred.detach().cpu().numpy()
        trend_forecast = trend_forecast.detach().cpu().numpy()
        seasonal_forecast = seasonal_forecast.detach().cpu().numpy()

        y_pred[i,:] = yscaler.inverse_transform(pred)
        trend[i,:] = yscaler.inverse_transform(trend_forecast)
        seasonality[i,:] = yscaler.inverse_transform(seasonal_forecast)

    return JSONResponse(status_code=200,
        content={
        "y_pred": y_pred, 
        "trend": trend, 
        "seasonality": seasonality
        }
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=50, reload=True)