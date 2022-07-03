# Contents of ~/my_app/main_page.py
import streamlit as st
from PIL import Image

st.title("Main page")
st.sidebar.markdown("# Main page")
st.markdown("## Project demo")

image = Image.open('static/sample.jpg')

st.image(image, caption='샘플 이미지..!')