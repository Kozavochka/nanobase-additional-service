import os
import pickle
import pathlib
import pandas as pd
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from dotenv import load_dotenv
from pg_connector import get_all_aloys_json, get_all_fuel_cells_json, get_all_composites_json
from rabbit_client import send_message
import json
from datetime import datetime

# Загрузить переменные из .env
load_dotenv()

app = FastAPI()
security = HTTPBasic()

# Получение данных пользователей из .env
users_db = {
    os.getenv("ADMIN_USERNAME"): os.getenv("ADMIN_PASSWORD"),
    os.getenv("USER_USERNAME"): os.getenv("USER_PASSWORD")
}

# Класс ModelWrapper
class ModelWrapper:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.name = None

    def set_models_data(self, model_reference, scaler_reference):
        self.model = model_reference
        self.scaler = scaler_reference

    def load(self, path):
        with open(path, 'rb') as handle:
            model_parts = pickle.load(handle)
            self.model = model_parts["model"]
            self.scaler = model_parts["scaler"]
            self.name = pathlib.Path(path).stem

    def predict(self, samples):
        preds = self.model.predict(samples)
        return preds

    def scaled_predict(self, samples):
        preds = self.predict(self.scaler.transform(samples))
        return preds

    def scaled_tranform_predict(self, samples: pd.DataFrame):
        new_samples = pd.DataFrame({
            "x_Co": samples["x_Co"].values,
            "x_Fe": samples["x_Fe"].values,
            "log_Po2": np.log10(samples["Po2"].values),
            "1000/T": 1000 / (samples["T"].values + 273),
            "1000/T_sint": 1000 / samples["T_sint"].values
        })
        return self.scaled_predict(new_samples)


# Загружаем модель
model_wrapper = ModelWrapper()
model_wrapper.load("LNCF_ETR_model.pickle")

def chunked(iterable, size):
    """Разбивает список на куски фиксированного размера."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

# Проверка авторизации
def authenticate(credentials: HTTPBasicCredentials):
    username = credentials.username
    password = credentials.password

    if users_db.get(username) != password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True

def default_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

@app.get("/health")
def health():
    return {"message": "OK"}

@app.post("/predict")
def predict(data: dict, credentials: HTTPBasicCredentials = Depends(security)):
    # Проверяем авторизацию
    authenticate(credentials)

    # Преобразование входных данных в DataFrame
    samples = pd.DataFrame(data)

    # Проверка входных данных
    required_columns = {"x_Co", "x_Fe", "Po2", "T", "T_sint"}
    if not required_columns.issubset(samples.columns):
        raise HTTPException(status_code=400, detail=f"Missing required columns: {required_columns - set(samples.columns)}")

    # Получение предсказания
    predictions = model_wrapper.scaled_tranform_predict(samples)
    return {"log(sigma)": predictions.tolist(), "sigma": (10 ** predictions).tolist()}

@app.get("/alloys")
def get_alloys(credentials: HTTPBasicCredentials = Depends(security)):
    authenticate(credentials)
    return get_all_aloys_json()

@app.get("/fuel-cells")
def get_alloys(credentials: HTTPBasicCredentials = Depends(security)):
    authenticate(credentials)
    return get_all_fuel_cells_json()

@app.post('/import-alloys')
def push_alloys(credentials: HTTPBasicCredentials = Depends(security)):
    authenticate(credentials)
    all_alloys = get_all_aloys_json()

    for chunk in chunked(all_alloys, 10):
        message = json.dumps(chunk, ensure_ascii=False, indent=2, default=default_serializer)
        send_message('alloys', message)

    return {"message": "OK"}


@app.post('/import-fuel-cells')
def push_alloys(credentials: HTTPBasicCredentials = Depends(security)):
    authenticate(credentials)
    data = get_all_fuel_cells_json()

    for chunk in chunked(data, 10):
        message = json.dumps(chunk, ensure_ascii=False, indent=2, default=default_serializer)
        send_message('fuel_cells', message)

    return {"message": "OK"}

@app.post('/import-all')
def push_alloys(credentials: HTTPBasicCredentials = Depends(security)):
    authenticate(credentials)
    data = get_all_fuel_cells_json()

    for chunk in chunked(data, 10):
        message = json.dumps(chunk, ensure_ascii=False, indent=2, default=default_serializer)
        send_message('fuel_cells', message)

    data = get_all_aloys_json()

    for chunk in chunked(data, 10):
        message = json.dumps(chunk, ensure_ascii=False, indent=2, default=default_serializer)
        send_message('alloys', message)

    data = get_all_composites_json()

    for chunk in chunked(data, 10):
        message = json.dumps(chunk, ensure_ascii=False, indent=2, default=default_serializer)
        send_message('composite', message)

    return {"message": "OK"}