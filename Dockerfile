# Используем базовый образ с Python
FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt и устанавливаем зависимости
COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install -y gcc libpq-dev libgl1  libglib2.0-0

RUN pip install --no-cache-dir --timeout=100 -r requirements.txt


# Копируем приложение
COPY . .

# Убедимся, что файл модели добавлен
COPY LNCF_ETR_model.pickle LNCF_ETR_model.pickle

# Запуск API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
