# Contents of ~/my_app/main_page.py
import streamlit as st
from PIL import Image

# st.sidebar.markdown("# Main page")
# st.markdown("## Project demo")

st.title("biTocoin")
st.write("credible & explainable forecast service")

image = Image.open('static/images/sample.jpg')
st.image(image)

col1, col2 = st.columns([1,1])

col1.markdown("## Motivation")
col1.write("설명 추가")

col2.markdown("## Dataset")
col2.write("설명 추가")

col1.markdown("## Pattern Recognization")
col1.write("설명 추가")

col2.markdown("## Forcasting prices")
col2.write("설명 추가")

# st.markdown("## Motivation")
# st.write("설명 추가")

# st.markdown("## Dataset")
# st.write("설명 추가")

# st.markdown("## Pattern Recognization")
# st.write("설명 추가")

# st.markdown("## Forcasting prices")
# st.write("설명 추가")

st.write("---")
st.markdown("## Members")
col1, col2, col3 = st.columns([1,1,1])
