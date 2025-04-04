# Используем базовый образ с Python
FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы requirements.txt и устанавливаем зависимости
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Копируем всё приложение в контейнер
COPY . .

# Убедимся, что файл модели добавлен
COPY iris_model.joblib iris_model.joblib

# Команда для запуска приложения
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
