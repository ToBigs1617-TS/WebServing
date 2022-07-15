# Contents of ~/my_app/main_page.py
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu


# st.sidebar.markdown("# Main page")
# st.markdown("## Project demo")

st.title("BIGSCOIN [![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/ToBigs1617-TS/Bigscoin) ")
st.write("### Explainable Bitcoin Pattern Alert and Forecasting Service")


image = Image.open('static/images/bigscoin.png')
st.image(image, width=650)
st.write('')

col1, col2 = st.columns([1,1])

col1.markdown("## Motivation")
col1.write("주식 및 코인 투자자들은 차트에서 발생되는 패턴 중 매매하기 적합한 시점의 패턴을 관측하며 투자를 진행합니다. **BIGSCOIN은 매매에 적합한 패턴이 등장하는 것을 자동으로 감지하여** 투자자에게 유용한 정보를 제공합니다.")
st.write('')

col2.markdown("## Dataset")
col2.markdown("**Pyupbit 패키지**를 통해 2017년 8월부터 2022년 5월까지의 5분봉 시가, 고가, 저가, 종가 데이터를 추출하여 학습에 사용하였습니다.")
st.write('')

col1, col2 = st.columns([1,1])

col1.markdown("## Pattern Recognization")
col1.write("Rising wedge, falling wedge와 같은 매매에 적합한 패턴 5가지를 탐지하기 위해 2D CNN을 사용하여 분류를 시행합니다.")

col2.markdown("## Forcasting Prices")
col2.write("경향성(Trend)과 계절성(Seasonality)과 같은 시계열 특징을 분석하는 **해석 가능한 딥러닝 아키텍처인 Nbeats**를 학습하여 향후 4시간을 예측합니다.")


st.write("---")
st.write(" Developed By ToBigs 16&17 Time Series & XAI Team  \n  ###### 김권호 김상윤 김윤혜 김주호 김현태 나세연 박한나 유현우 이예림")

# st.markdown("## Members")

# col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])

# col1.write("##### 김권호")
# col2.write("##### 김상윤")
# col3.write("##### 김윤혜")
# col4.write("##### 김주호")
# col5.write("##### 김현태")

# col1.write("##### 나세연")
# col2.write("##### 박한나")
# col3.write("##### 유현우")
# col4.write("##### 이예림")


