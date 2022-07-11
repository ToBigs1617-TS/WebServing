import numpy as np
import pickle
import torch

# Preprocessing Upbit Data
def forecast_preprocess(df, yscaler):
    window_size = 144 # 모델 학습시 input_window 144으로 학습함.
    # yscaler = pickle.load(open('/Users/kim_yoonhye/Desktop/TS-컨퍼/github/WebServing/forecast/scaler.pkl', 'rb'))

    # data_tmp = pyupbit.get_ohlcv("KRW-BTC", interval="minute5", count=200)     # 5분봉 데이터
    # data_tmp.reset_index(inplace=True)

    # data_tmp.rename(columns={'index':'date'}, inplace=True)
    data_tmp = df[['Date', 'Close']]
    data_tmp.set_index('Date', inplace=True)

    data_tmp = data_tmp.iloc[-window_size:, :]
    # data_tmp.shape

    # scaling 
    tmp = yscaler.transform(np.array(data_tmp['Close']).reshape(-1, 1))
    data_tmp['Close'] = tmp

    Y = data_tmp.iloc[:, 0].values
    Y = torch.FloatTensor(Y)
    Y = Y.unsqueeze(-1)
    Y = Y.unsqueeze(0) # batch_size
    # print(Y.shape)

    return Y