import os
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from dotenv import load_dotenv
import joblib
import numpy as np
from pydantic import BaseModel

# Загрузить переменные из .env
load_dotenv()

app = FastAPI()
security = HTTPBasic()

# Получение данных пользователей из .env
users_db = {
    os.getenv("ADMIN_USERNAME"): os.getenv("ADMIN_PASSWORD"),
    os.getenv("USER_USERNAME"): os.getenv("USER_PASSWORD")
}

# Загрузка заранее обученной модели
model = joblib.load("iris_model.joblib")

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

# Определение структуры входных данных
class InputData(BaseModel):
    features: list[float]

@app.get("/health")
def get_status():
    return {"message": "OK"}

@app.post("/predict")
def predict(input_data: InputData, credentials: HTTPBasicCredentials = Depends(security)):
    # Проверяем авторизацию
    authenticate(credentials)

    # Преобразование входных данных в массив numpy
    features = np.array(input_data.features).reshape(1, -1)

    # Проверяем соответствие числа признаков
    if features.shape[1] != 4:  # Убедитесь, что размер соответствует модели (4 признака для Iris)
        raise HTTPException(status_code=400, detail="Invalid input size. Expected 4 features.")

    # Получение предсказания
    prediction = model.predict(features)
    return {"predicted_class": int(prediction[0])}
