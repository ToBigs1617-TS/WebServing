# Bigscoin - Web Demo
> ## Explainable Bitcoin Pattern Alert and Forecasting Service

## File Directory

```
.
├── frontend
│   ├── main_page.py
│   ├── pages
│   │   └── model_page.py
│   └── images
│  
├── backend
│   └── main.py
│  
├── forecast
│   ├── model.py
│   ├── preprocess.py
│   ├── utils.py
│   ├── model.pt
│   └── scaler.pkl
│    
└── recognition
    └── Model.pt

```

# Usage
## 0. Create virtual environment
서비스를 실행하기 위한 가상환경을 생성합니다.     
python >= 3.8.1 
```
pyenv virtualenv bigscoin-web
```
## 1. Installation
```
pyenv activate bigscoin-web
```
```
pip install -r requirements.txt
```
## 2. run backend & frontend
### backend
```
python backend/main.py
```
### frontend
```
streamlit run frontend/main_page.py
```
_서버 배포 후 수정 + Demo  예정_


# Demo - _이미지 더 나은 걸로 수정할 예정_
## Pattern Recognition
![image](https://user-images.githubusercontent.com/56261032/179201735-bde0d2fa-b362-46a2-b07c-883c0bb58bad.png)


## Forecasting Bitcoin Price
![image](https://user-images.githubusercontent.com/56261032/179201568-cfd22417-06fc-472d-b251-fa17eda5d3ff.png)

