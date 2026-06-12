# peroxite

FastAPI-сервис для:
- предсказания проводимости по модели `LNCF_ETR_model.pickle`;
- работы с данными из Postgres;
- отправки сообщений в RabbitMQ;
- обработки и оцифровки графиков;
- работы с S3-совместимым хранилищем;
- распознавания осей и тиков через YOLO.

## Требования

- Python 3.10+.
- Доступ к Postgres, RabbitMQ и S3-совместимому хранилищу, если используются соответствующие эндпоинты.
- Наличие файлов моделей и вспомогательных артефактов, описанных ниже.

## Установка

### Локально

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Через Docker

```bash
docker compose up --build
```

Сервис поднимается на `http://localhost:5000`.

## Настройка `.env`

Проект читает переменные окружения через `python-dotenv`. Файл `.env` должен лежать в корне репозитория.

### Обязательные переменные для авторизации

Используются для HTTP Basic Auth в `main.py`:

```env
ADMIN_USERNAME=admin
ADMIN_PASSWORD=change_me
USER_USERNAME=user
USER_PASSWORD=change_me
```

### Postgres

Используются в `pg_connector.py`:

```env
PG_HOST=localhost
PG_PORT=5432
PG_DB_NAME=database_name
PG_USER=postgres
PG_PASSWORD=postgres_password
```

Подключение создаётся с `sslmode=require`.

### RabbitMQ

Используются в `rabbit_client.py`:

```env
RABBIT_MQ_HOST=localhost
RABBIT_MQ_PORT=5672
RABBIT_MQ_VHOST=/
RABBIT_MQ_USER=guest
RABBIT_MQ_PASSWORD=guest
```

### S3 / Object Storage

Используются в `config.py` и `s3_service.py`:

```env
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
AWS_BUCKET=your-bucket-name
AWS_ENDPOINT_URL=https://your-s3-endpoint
```

`AWS_ENDPOINT_URL` можно не задавать, если используется стандартный AWS S3.

### YOLO и OCR

Используются в `yolo_service.py`. Если не заданы, берутся значения по умолчанию из папки `YOLO/models/...`:

```env
YOLO_AXIS_MODEL_PATH=YOLO/models/axis/best.pt
YOLO_TICKS_MODEL_PATH=YOLO/models/ticks/best.pt
EASYOCR_LANGS=en
EASYOCR_GPU=0
EASYOCR_ALLOWLIST=0123456789.,-+eE
```

## Обязательные файлы и модели

Для корректной работы проекта в репозитории должны быть:

```text
LNCF_ETR_model.pickle
YOLO/models/axis/best.pt
YOLO/models/ticks/best.pt
```

### Назначение файлов

- `LNCF_ETR_model.pickle` — сериализованная модель для эндпоинта `/predict`.
- `YOLO/models/axis/best.pt` — модель для `/yolo-axis`.
- `YOLO/models/ticks/best.pt` — модель для `/yolo-ticks`.

Если вы хотите использовать другие пути, укажите их через:
- `YOLO_AXIS_MODEL_PATH`
- `YOLO_TICKS_MODEL_PATH`

## Сборка и запуск

### Локальный запуск

```bash
uvicorn main:app --host 0.0.0.0 --port 5000
```

### Docker

```bash
docker compose up --build
```

## Что должно быть в проекте

Минимально необходимая структура:

```text
.
├── main.py
├── config.py
├── pg_connector.py
├── rabbit_client.py
├── s3_service.py
├── chart_processor.py
├── calibration_service.py
├── colorref_service.py
├── mask_service.py
├── property_relation_service.py
├── yolo_service.py
├── fit_model_service.py
├── heatmap_service.py
├── clustering_service.py
├── curve_features_service.py
├── groupingml_service.py
├── LNCF_ETR_model.pickle
├── YOLO/
│   └── models/
│       ├── axis/
│       │   └── best.pt
│       └── ticks/
│           └── best.pt
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env
```

## Проверка работоспособности

После запуска проверьте:

```bash
curl http://localhost:5000/health
```

Ожидаемый ответ:

```json
{"message":"OK"}
```

## Основные эндпоинты

- `GET /health`
- `POST /predict`
- `GET /alloys`
- `GET /fuel-cells`
- `POST /import-alloys`
- `POST /import-fuel-cells`
- `POST /import-all`
- `POST /process_chart`
- `POST /db-scan-process-chart`
- `POST /colorref-process-chart`
- `POST /evaluate-mask`
- `POST /relation`
- `POST /yolo-axis`
- `POST /yolo-ticks`

Дополнительные аналитические эндпоинты:
- `POST /analysis/fit-model`
- `POST /analysis/heatmap`
- `POST /analysis/curve-features`
- `POST /analysis/clustering`
- `POST /spline-interpolate`

## Примечания

- В коде некоторые сервисы требуют доступ к внешней инфраструктуре: Postgres, RabbitMQ и S3.
- Если вы запускаете только `/health` и `/predict`, достаточно модели `LNCF_ETR_model.pickle` и корректных переменных Basic Auth.
- Если используете YOLO-эндпоинты, убедитесь, что установлен `ultralytics`, а файлы `best.pt` доступны по указанным путям.
