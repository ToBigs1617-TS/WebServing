# Contents of ~/my_app/pages/page_2.py
import streamlit as st
from streamlit_autorefresh import st_autorefresh

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import cufflinks as cf
import pyupbit
import mpl_finance
from tensorflow import keras

import requests
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from forecast.model import *
from utils import *


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
    st.write("Rising Wedge, Falling Wedge와 같은 매매에 적합한 패턴 5가지를 탐지하기 위해 2D CNN을 사용하여 분류를 시행합니다.")
    
    # Pattern Explanation Expander
    with st.expander("pattern explanation"):
        _, col2, _, col4, _ = st.columns([1,1,1,1,1])
        
        # col2.markdown("#### Ascending Triangle")
        col2.markdown("<h4 style='text-align: center;'>Ascending Triangle</h4>", unsafe_allow_html=True)
        col2.image("./static/images/ascending_triangle.png")

        # col4.markdown("#### Rising Wedge")
        col4.markdown("<h4 style='text-align: center;'>Rising Wedge</h4>", unsafe_allow_html=True)
        col4.image("./static/images/rising_wedge.png")
        
        # col2.markdown("#### Descending Triangle")
        col2.markdown("<h4 style='text-align: center;'>Descending Triangle</h4>", unsafe_allow_html=True) # can select color - ex)color: black;
        col2.image('./static/images/descending_triangle.png')
        
        # col4.markdown("#### Falling Wedge")
        col4.markdown("<h4 style='text-align: center;'>Falling Wedge</h4>", unsafe_allow_html=True)
        col4.image('./static/images/falling_wedge.png')
        
        # col2.markdown("#### Symmetric Triangle")
        col2.markdown("<h4 style='text-align: center;'>Symmetric Triangle</h4>", unsafe_allow_html=True)
        col2.image('./static/images/symmetrical_triangle.png')
    
    # Candlestick Chart 그리기
    df2 = df.copy()
    df2.set_index('Date',inplace=True)
    qf = cf.QuantFig(df2, legend='top', name='BTC')
    
    qf.add_rsi(periods=5, color='java')

    qf.add_bollinger_bands(periods=5,boll_std=2,colors=['magenta','grey'],fill=True)

    qf.add_volume()
    fig = qf.iplot(asFigure=True, dimensions=(800, 600))


    with st.spinner("In progress.."):
        st.subheader("Bitcoin Prices")
        st.plotly_chart(fig, use_container_width=True)

        pattern_pred, heatmap = get_recognition()
    
        # Grad-CAM
        # st.subheader("XAI")
    
        # fig2 = plt.figure(figsize=(30,4))
        # plt.imshow(heatmap)
        # plt.axis('off')
        # st.pyplot(fig2)

        ## Grad-CAM
        # 전체 candlestick image (200개)
        fig = plt.figure(figsize=(30,10))
        ax = fig.add_subplot(111)
        plt.xlim(-20, 220)
        mpl_finance.candlestick2_ohlc(ax, df['Open'], df['High'], df['Low'], df['Close'], width=0.5, colorup='r', colordown='b')
        
        # 마지막 60개 candlestick image
        fig2 = plt.figure(figsize=(30,10))
        ax2 = fig2.add_subplot(111)
        mpl_finance.candlestick2_ohlc(ax2, df[-60:]['Open'], df[-60:]['High'], df[-60:]['Low'], df[-60:]['Close'], width=0.5, colorup='r', colordown='b')
        fig2.canvas.draw()
        ary_60 = np.array(fig2.canvas.renderer._renderer)

        # 앞 140개 candlestick image
        fig3 = plt.figure(figsize=(30,10))
        ax3 = fig3.add_subplot(111)
        plt.xlim(0, 207)
        plt.axis('off')
        mpl_finance.candlestick2_ohlc(ax3, df[:140]['Open'], df[:140]['High'], df[:140]['Low'], df[:140]['Close'], width=0.5, colorup='r', colordown='b')
        fig3.canvas.draw()
        ary_140 = np.array(fig3.canvas.renderer._renderer)

        xai_60 = keras.preprocessing.image.array_to_img(ary_60)
        w, h = xai_60.size

        # heatmap -> array
        fig4 = plt.figure(figsize=(10,10))
        plt.imshow(heatmap)
        plt.xlim(-230, 110)
        plt.ylim(-130, 200)
        plt.axis('off')

        fig4.canvas.draw()
        heatmap_ary = np.array(fig4.canvas.renderer._renderer)
        heatmap = keras.preprocessing.image.array_to_img(heatmap_ary)
        heatmap = heatmap.resize((w, h))

        heatmap = keras.preprocessing.image.img_to_array(heatmap)

        superimposed_img = heatmap + ary_140
        superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
        st.subheader("XAI: Grad-CAM")
        st.image(superimposed_img)

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

    # Raw Data
    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(df)

    st.write('---')

    ### Forecasting Stock Prices
    st.header("Forecasting Bitcoin Price")
    st.write("경향성(Trend)과 계절성(Seasonality)과 같은 시계열 특징을 분석하는 해석 가능한 딥러닝 아키텍처인 Nbeats를 학습하여 향후 4시간을 예측합니다.")

    with st.spinner("forecasting..."):
        get_forecast()

if __name__ == "__main__":
    main()