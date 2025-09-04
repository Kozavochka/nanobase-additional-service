# Используем базовый образ с Python
FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt и устанавливаем зависимости
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Копируем приложение
COPY . .

# Убедимся, что файл модели добавлен
COPY LNCF_ETR_model.pickle LNCF_ETR_model.pickle

# Запуск API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
