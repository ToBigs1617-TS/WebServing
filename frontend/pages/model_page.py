# Contents of ~/my_app/pages/page_2.py
import streamlit as st
from streamlit_autorefresh import st_autorefresh

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import cufflinks as cf
import pyupbit
import requests
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from forecast.model import *
from forecast.utils import *


backend_address = "http://0.0.0.0:50"
window_size = 144
yscaler = pickle.load(open('/Users/kim_yoonhye/Desktop/TS-컨퍼/github/WebServing/forecast/scaler.pkl', 'rb'))

st.set_page_config(page_icon='📈' ,layout="wide")

# start_date = st.sidebar.date_input('please select a start date', datetime(2022, 5, 4))
# # end_date = st.sidebar.date_input('End date', datetime(2022, 7, 4))
# end_date = datetime.now()

# load data
df = pyupbit.get_ohlcv("KRW-BTC", interval="minute5", count=200)     # 5분봉 데이터
df.reset_index(inplace=True)
df.rename(columns={'index':'Date', 'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume', 'value':'Value'}, inplace=True)
true = df['Close'].values.reshape(-1, 1)


def get_recognition():
    response = requests.get(
        url=f"{backend_address}/recognition", 
        )
    pred = response.json()['pred']
    heatmap = response.json()['heatmap']
    return pred, heatmap

def get_forecast():

    data_tmp = df[['Date', 'Close']]
    data_tmp.set_index('Date', inplace=True)
    data_tmp = data_tmp.iloc[-window_size:, :]

    response = requests.get(
        url=f"{backend_address}/forecast", 
        )
 
    y_pred = response.json()['y_pred']
    trend = response.json()['trend']
    seasonality = response.json()['seasonality']
    
    fig = plot_prediction(true, y_pred, trend, seasonality, data_tmp)
    st.plotly_chart(fig, use_container_width=True)
    return y_pred, trend, seasonality


def main():
    # 5분 간격으로 새로고침. 약 8시간 동안 auto-fresh
    st_autorefresh(interval=300000, limit=100, key="counter")

    st.title("Model page") # 페이지 제목 뭐로 하죠??
    st.sidebar.markdown("# Model page ")
    # st.sidebar.header('Menu')

    # Ticker / Startdate 설정
    # st.sidebar.subheader('Ticker')
    # ticker = st.sidebar.selectbox('please select a ticker', ['BTC', 'ETH', 'USDT'])
    # st.sidebar.subheader('Start Date')

    # Select patterns to receive alarms 
    st.sidebar.subheader("Select patterns")
    st.sidebar.write("알림 받을 패턴을 선택해주세요. (다중 선택 가능)")
    rising_wedge = st.sidebar.checkbox('Rising Wedge')
    falling_wedge = st.sidebar.checkbox('Falling Wedge')
    ascending_triangle = st.sidebar.checkbox('Ascending Triangle')
    descending_triangle = st.sidebar.checkbox('Descending Triangle')
    symmetric_triangle = st.sidebar.checkbox('Symmetric Triangle')

    num_patterns = rising_wedge + falling_wedge + ascending_triangle + descending_triangle + symmetric_triangle
    if not num_patterns:
        st.sidebar.error("패턴을 한 개 이상 선택해주세요.")
        
    ### Pattern Recognition 
    st.header("Pattern Recognition")
    st.write("현재 시점을 기준으로 가장 유사한 패턴을 감지합니다.")
    
    # Pattern Explanation Expander
    with st.expander("pattern explanation"):
        col1, col2 = st.columns([1,1])
        _, ccol2, _, ccol4, _ = st.columns([1,1,1,1,1])
        
        col1.markdown("<h4 style='text-align: center;'>Ascending Triangle</h4>", unsafe_allow_html=True)
        ccol2.image("./static/images/ascending_triangle.png")

        col2.markdown("<h4 style='text-align: center;'>Rising Wedge</h4>", unsafe_allow_html=True)
        ccol4.image("./static/images/rising_wedge.png")

        col1, col2 = st.columns([1,1])
        _, ccol2, _, ccol4, _ = st.columns([1,1,1,1,1])

        col1.markdown("<h4 style='text-align: center;'>Descending Triangle</h4>", unsafe_allow_html=True) # can select color - ex)color: black;
        ccol2.image('./static/images/descending_triangle.png')
        
        col2.markdown("<h4 style='text-align: center;'>Falling Wedge</h4>", unsafe_allow_html=True)
        ccol4.image('./static/images/falling_wedge.png')
        
        col1, col2 = st.columns([1,1])
        _, ccol2, _, ccol4, _ = st.columns([1,1,1,1,1])
        
        col1.markdown("<h4 style='text-align: center;'>Symmetric Triangle</h4>", unsafe_allow_html=True)
        ccol2.image('./static/images/symmetrical_triangle.png')
    
    # Candlestick Chart 그리기
    df2 = df.copy()
    df2.set_index('Date',inplace=True)
    qf = cf.QuantFig(df2, legend='top', name='BTC')
    
    qf.add_rsi(periods=5, color='java')

    qf.add_bollinger_bands(periods=5,boll_std=2,colors=['magenta','grey'],fill=True)

    qf.add_volume()
    fig = qf.iplot(asFigure=True, dimensions=(800, 600))


    with st.spinner("In progress.."):
        col1, col2 = st.columns([3,1])
        col1.subheader("Bitcoin Prices")
        col1.plotly_chart(fig, use_container_width=True)

        pattern_pred, heatmap = get_recognition()

        # Alert Box
        thresh = 0.8
        # st.write(pattern_pred[0][3])
        if rising_wedge == True and pattern_pred[0][3] >= thresh:
            st.warning(f"**Rising Wedge** 패턴과 가장 유사합니다.")
        if falling_wedge == True and pattern_pred[0][2] >= thresh:
            st.warning(f"**Falling Wedge** 패턴과 가장 유사합니다.")
        if ascending_triangle == True and pattern_pred[0][0] >= thresh:
            st.warning(f"**Ascending Triangle** 패턴과 가장 유사합니다.")
        if descending_triangle == True and pattern_pred[0][1] >= thresh:
            st.warning(f"**Descending Triangle** 패턴과 가장 유사합니다.")
        if symmetric_triangle == True and pattern_pred[0][4] >= thresh:
            st.warning(f"**Symmetric Triangle** 패턴과 가장 유사합니다.")
    
        # Grad-CAM
        col2.subheader("XAI")
    
        fig2 = plt.figure(figsize=(30,4))
        plt.imshow(heatmap)
        plt.axis('off')
        col2.pyplot(fig2)


    # Raw Data
    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(df)

    st.write('---')

    ### Forecasting Stock Prices
    st.header("Forecasting Bitcoin Price")
    st.write("현시점부터 4시간 후까지의 종가를 예측합니다.")

    with st.spinner("forecasting..."):
        get_forecast()

if __name__ == "__main__":
    main()