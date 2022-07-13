from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
import datetime

import uvicorn
import torch
import tensorflow as tf
import sys
import os
import pyupbit

# from pandera.typing.fastapi import UploadFile
# from pandera.typing import DataFrame


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from forecast.model import *
from forecast.preprocess import forecast_preprocess

app = FastAPI()

# class CPU_Unpickler(pickle.Unpickler):
#     def find_class(self, module, name):
#         if module == 'torch.storage' and name == '_load_from_bytes':
#             return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
#         else: return super().find_class(module, name)

#contents = pickle.load(f) becomes...
# contents = CPU_Unpickler(f).load()
# yscaler = CPU_Unpickler(open('/Users/kim_yoonhye/Desktop/TS-컨퍼/github/WebServing/forecast/scaler.pkl', 'rb')).load()



yscaler = pickle.load(open('/Users/kim_yoonhye/Desktop/TS-컨퍼/github/WebServing/forecast/scaler.pkl', 'rb'))


# class data_check(BaseModel):
# #     # date: float = Field(None)
#     Date: datetime = Field(None)
#     Close: float 

    # class Config:
    #     orm_mode = True

def load_recognition_model():
    model = tf.keras.models.load_model('/Users/kim_yoonhye/Desktop/TS-컨퍼/github/WebServing/recognition/sample_model')
    return model

def load_forecast_model():
    model = NBeatsNet(stack_types=(NBeatsNet.TREND_BLOCK, NBeatsNet.SEASONALITY_BLOCK),
                        forecast_length=48,
                        backcast_length=144,
                        hidden_layer_units=128,
                        )
    model.load_state_dict(torch.load('/Users/kim_yoonhye/Desktop/TS-컨퍼/github/WebServing/forecast/model.pth', map_location=torch.device('cpu')))
    return model


@app.post("/file")
def upload_file(file: UploadFile = File(...)):
    d_file = file.to_dict()
    return d_file
    read_file = file.read()
    
    # pandas 모듈을 사용하여 업로드한 파일을 pandas.Dataframe 형식으로 가져온다.
    excel_file = pd.read_excel(read_file)
    
    # Dataframe 인스턴스를 json 으로 각 row마다 나누어 리스트화 시킨다.
    json_data = excel_file.to_json(orient='records')
    
    # raw한 json데이터를 읽는다.
    data = json.loads(json_data)
    
    # 각 row 데이터에 해당하는 json을 테이블에 매핑 후 bulk_create한다.
    for row in data:
        model = TableModel(**row)
        db.add(model)
        
    db.commit()
    return

@app.get("/recognition")
def get_recognition():
    ### 01. 데이터 load & 입력 전처리 코드
    df = pyupbit.get_ohlcv("KRW-BTC", interval="minute5", count=60)
    df = df[['close']]
    df = df.to_numpy()
    df = df.reshape(-1, 60, 1)

    model = load_recognition_model()
    pred = model.predict(df) # 각 패턴에 대한 확률값 반환

    return JSONResponse(
        content={
            "pred": pred.tolist()
        }
    )
@app.get("/forecast")
# def get_forecast(data: data_check):
def get_forecast():
    df = pyupbit.get_ohlcv("KRW-BTC", interval="minute5", count=200)     # 5분봉 데이터
    df.reset_index(inplace=True)
    df.rename(columns={'index':'Date', 'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume', 'value':'Value'}, inplace=True)
    data_tmp = df[['Date', 'Close']]
    data_tmp.set_index('Date', inplace=True)

    window_size = 144
    data_tmp = data_tmp.iloc[-window_size:, :]
    # data = data_tmp.to_json(orient="records")

    # print(df.dtypes)
    # print(df.Date)
    # df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    # print(df.dtypes)
    # print(type(data))
    # print(data)
    # data = data_tmp.to_dict()
    # df = pd.read_json(data_tmp)
    
    # return data_tmp
    # df = json.loads(data)
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