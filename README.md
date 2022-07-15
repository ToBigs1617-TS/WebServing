# Bigscoin - Web Demo
> ## Explainable Bitcoin Pattern Alert and Forecasting Service

## File Directory

```
.
├── frontend
│   ├── main_page.py
│   └── pages
│       └── model_page.py
│   
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


# Demo
## Pattern Recognition
<img width="958" alt="image" src="https://user-images.githubusercontent.com/56261032/179317219-20da2ccc-c2bc-44fb-b16a-074de1039471.png">
<img width="960" alt="image" src="https://user-images.githubusercontent.com/56261032/179317578-ac839c46-07d1-4698-a0bd-87e5b540236e.png">

## Forecasting Bitcoin Price
<img width="945" alt="image" src="https://user-images.githubusercontent.com/56261032/179317333-de1e1cf4-88b6-4bb0-989e-8d9a3c69354c.png">

