from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import uvicorn
import torch
import tensorflow as tf
import pyupbit
import mpl_finance

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from forecast.model import *
from forecast.preprocess import forecast_preprocess
from utils import make_gradcam_heatmap, show_heatmap

app = FastAPI()

yscaler = pickle.load(open('/Users/kim_yoonhye/Desktop/TS-컨퍼/github/WebServing/forecast/scaler.pkl', 'rb'))


def load_recognition_model():
    model = tf.keras.models.load_model('../recognition/Model.pt')
    return model

def load_forecast_model():
    model = NBeatsNet(stack_types=(NBeatsNet.TREND_BLOCK, NBeatsNet.SEASONALITY_BLOCK),
                        forecast_length=48,
                        backcast_length=144,
                        hidden_layer_units=128,
                        )
    model.load_state_dict(torch.load('/Users/kim_yoonhye/Desktop/TS-컨퍼/github/WebServing/forecast/model.pth', map_location=torch.device('cpu')))
    return model


@app.get("/recognition")
def get_recognition():
    ## 01. 데이터 load
    df = pyupbit.get_ohlcv("KRW-BTC", interval="minute5", count=60)
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    mpl_finance.candlestick2_ohlc(ax, df['open'], df['high'], df['low'], df['close'], width=0.5, colorup='r', colordown='b')
    fig.canvas.draw()
    X =np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    X =X.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    X = np.rint(X[:,:,:3] @ [0.2126, 0.7152, 0.0722]).astype(np.uint8)
    print(X.shape)
    X = np.reshape(X,(1,800,400,1))
    print(X.shape)
    model = load_recognition_model()
    pred = model.predict(X) # 각 패턴에 대한 확률값 반환
    
    heatmap = show_heatmap(model, X)

    return JSONResponse(
        content={
            "pred": pred.tolist(),
            "heatmap": heatmap.tolist()
        }
    )


@app.get("/forecast")
def get_forecast():
    df = pyupbit.get_ohlcv("KRW-BTC", interval="minute5", count=200)     # 5분봉 데이터
    df.reset_index(inplace=True)
    df.rename(columns={'index':'Date', 'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume', 'value':'Value'}, inplace=True)
    data_tmp = df[['Date', 'Close']]
    data_tmp.set_index('Date', inplace=True)

    window_size = 144
    data_tmp = data_tmp.iloc[-window_size:, :]

    forecast_df = forecast_preprocess(data_tmp, yscaler)
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
        # seasonality[i,:] = yscaler.inverse_transform(seasonal_forecast)
        seasonality[i,:] = y_pred[i,:] - trend[i,:]

    return JSONResponse(
        content={
        "y_pred": y_pred.tolist(), 
        "trend": trend.tolist(), 
        "seasonality": seasonality.tolist()
        }
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=50, reload=True)