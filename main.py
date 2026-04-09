import os
import pickle
import pathlib
import pandas as pd
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, status, Body, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from dotenv import load_dotenv
from pg_connector import get_all_aloys_json, get_all_fuel_cells_json, get_all_composites_json
from rabbit_client import send_message
import json
from datetime import datetime
from s3_service import S3Service
from calibration_service import ROI, AxisConfig, PlotCalibrator
from chart_processor import ChartProcessorService
from typing import Dict, Any, List, Optional
from colorref_service import ColorRefService
from groupingml_service import GroupingMLService
from property_relation_service import PropertyRelation
from pydantic import BaseModel, ValidationError
from mask_service import MaskService
from fit_model_service import (
    DEFAULT_COMPLEXITY_PENALTY_WEIGHT,
    DEFAULT_DUPLICATE_X_POLICY,
    DEFAULT_GRID_SIZE,
    DEFAULT_RETURN_ALL_CANDIDATES,
    DEFAULT_ROBUST_CLEANING,
    DEFAULT_SELECTION_METRIC,
    DEFAULT_TIE_THRESHOLD,
    FitModelError,
    fit_model_series,
)

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


def safe_ticks(vmin, vmax, step):
    if not step:
        return None
    try:
        step = float(step)
    except Exception:
        return None
    if step == 0:
        return None
    # проверяем направление
    if (vmax > vmin and step < 0) or (vmax < vmin and step > 0):
        step = -step
    # np.arange работает с float
    ticks = list(np.arange(vmin, vmax + step, step))
    return ticks


def build_calibrator_from_calibration(calibration: Dict[str, Any]) -> PlotCalibrator:
    required_fields = (
        "x1_pix",
        "x2_pix",
        "y1_pix_axis",
        "y2_pix_axis",
        "x1_val",
        "x2_val",
        "y1_val",
        "y2_val",
    )

    missing = [field for field in required_fields if field not in calibration]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing calibration fields: {', '.join(missing)}",
        )

    try:
        x1_pix = float(calibration["x1_pix"])
        x2_pix = float(calibration["x2_pix"])
        y1_pix_axis = float(calibration["y1_pix_axis"])
        y2_pix_axis = float(calibration["y2_pix_axis"])
        x1_val = float(calibration["x1_val"])
        x2_val = float(calibration["x2_val"])
        y1_val = float(calibration["y1_val"])
        y2_val = float(calibration["y2_val"])
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Calibration fields must be numeric: {exc}",
        ) from exc

    if x1_val == x2_val:
        raise HTTPException(
            status_code=400,
            detail="Invalid calibration: x-axis range is zero (x1_val == x2_val).",
        )
    if y1_val == y2_val:
        raise HTTPException(
            status_code=400,
            detail="Invalid calibration: y-axis range is zero (y1_val == y2_val).",
        )

    roi = ROI(
        x_left=min(x1_pix, x2_pix),
        x_right=max(x1_pix, x2_pix),
        y_top=min(y1_pix_axis, y2_pix_axis),
        y_bottom=max(y1_pix_axis, y2_pix_axis),
    )
    xaxis = AxisConfig(vmin=x1_val, vmax=x2_val)
    yaxis = AxisConfig(vmin=y1_val, vmax=y2_val)

    try:
        return PlotCalibrator(roi, xaxis, yaxis)
    except AssertionError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid calibration: {exc}") from exc

# Методы графической детекции
@app.post("/process_chart")
def process_chart(markup: dict = Body(...), calibration: dict = Body(...), image_path: str = Body(...)):
    # 1. Валидируем калибровку и создаем калибратор
    calibrator = build_calibrator_from_calibration(calibration)

    # 2. Инициализируем сервисы
    s3 = S3Service()
    processor = ChartProcessorService(s3, calibrator)

    # 3. Запускаем обработку
    result = processor.process(image_path=image_path, markup=markup, calibration=calibration)

    return result


@app.post("/db-scan-process-chart")
def db_scan_process_chart(
    markup: Dict[str, Any] = Body(...),
    calibration: Dict[str, Any] = Body(...),
    image_path: str = Body(...),
    eps_setting: float = Body(15.0)  # 8 (близко) или 15 (средне)
):
    # 1) Валидируем калибровку и создаем калибратор
    calibrator = build_calibrator_from_calibration(calibration)

    # 2) Инициализируем сервисы
    s3 = S3Service()
    processor = ChartProcessorService(s3, calibrator)

    # 3) Запуск новой логики: DBSCAN по цвету
    result = processor.process_dbscan(
        image_path=image_path,
        markup=markup,
        calibration=calibration,
        eps_setting=float(eps_setting)  # 8.0 или 15.0
    )

    return result



@app.post("/colorref-process-chart")
# async def colorref_process_chart(payload: dict = Body(...)):
#     print(payload)
#     return payload
async def colorref_process_chart(
    markup: Dict[str, Any] = Body(...),
    calibration: Dict[str, Any] = Body(...),
    image_path: str = Body(...),
    refs: list[Dict[Any, Any]] = Body(...),   # список {name:..., rgb:...}
):
    s3 = S3Service()
    service = ColorRefService(s3)
    result = await service.process(
        markup=markup,
        calibration=calibration,
        image_path=image_path,
        refs=refs,
    )
    return result


@app.post("/groupingml-process-chart")
def groupingml_process_chart(
    calibration: Dict[str, Any] = Body(...),
    image_path: str = Body(...),
    clusters: Optional[Dict[str, List[List[int]]]] = Body(None),
    chartreader_groups: Optional[List[List[float]]] = Body(None),
    cluster_pixel_values: Optional[Dict[str, List[List[int]]]] = Body(None),
    noise: Optional[List[List[int]]] = Body(None),
    category_idx: int = Body(0),
    post_merge_enabled: bool = Body(True), #Параметр merge кластеров
    post_merge_distance_px: float = Body(18.0),
):
    resolved_clusters = clusters or {}
    resolved_from = "clusters"

    if not resolved_clusters and chartreader_groups:
        resolved_clusters = GroupingMLService.chartreader_groups_to_clusters(
            chartreader_groups=chartreader_groups,
            category_idx=category_idx,
        )
        resolved_from = "chartreader_groups"

    if not resolved_clusters:
        raise HTTPException(status_code=400, detail="Clusters payload is empty")

    if post_merge_enabled:
        resolved_clusters = GroupingMLService.merge_clusters_by_endpoint_distance(
            clusters=resolved_clusters,
            post_merge_distance_px=post_merge_distance_px,
        )

    calibrator = build_calibrator_from_calibration(calibration)

    s3 = S3Service()
    service = GroupingMLService(s3, calibrator)
    result = service.process(
        image_path=image_path,
        clusters=resolved_clusters,
        cluster_pixel_values=cluster_pixel_values,
        noise=noise,
    )
    result["resolved_from"] = resolved_from
    result["category_idx"] = category_idx
    result["post_merge_enabled"] = post_merge_enabled
    result["post_merge_distance_px"] = post_merge_distance_px
    return result


class RelationInput(BaseModel):
    T_A: List[float]
    A: List[float]
    T_B: List[float]
    B: List[float]
    method: str = "cubic"  # по умолчанию кубический сплайн

class SplineInterpolationInput(BaseModel):
    x: List[float]
    y: List[float]
    n_points: int = 200
    method: str = "cubic"  # cubic|linear


class FitPoint(BaseModel):
    x: Optional[float]
    y: Optional[float]


class FitMetricsResponse(BaseModel):
    rmse: float
    mae: float
    r2: float


class FitWarningResponse(BaseModel):
    code: str
    message: str


class FitCandidateResponse(BaseModel):
    name: str
    status: str
    params: Optional[Dict[str, float | str]] = None
    metrics: Optional[FitMetricsResponse] = None
    score: Optional[float] = None
    complexity_rank: Optional[int] = None
    fit_points_used: Optional[int] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class FitCleaningReportResponse(BaseModel):
    input_points: int
    removed_invalid_points: int
    removed_outliers: int
    duplicate_x_groups: int
    points_after_cleaning: int
    unique_x_after_cleaning: int


class FitModelRequest(BaseModel):
    series: List[FitPoint]
    candidate_models: Optional[List[str]] = None
    selection_metric: str = DEFAULT_SELECTION_METRIC
    grid_size: int = DEFAULT_GRID_SIZE
    robust_cleaning: bool = DEFAULT_ROBUST_CLEANING
    duplicate_x_policy: str = DEFAULT_DUPLICATE_X_POLICY
    complexity_penalty_weight: float = DEFAULT_COMPLEXITY_PENALTY_WEIGHT
    tie_threshold: float = DEFAULT_TIE_THRESHOLD
    return_all_candidates: bool = DEFAULT_RETURN_ALL_CANDIDATES


class FitModelResponse(BaseModel):
    best_model: FitCandidateResponse
    candidates: List[FitCandidateResponse]
    curve_points: List[FitPoint]
    cleaning_report: FitCleaningReportResponse
    warnings: List[FitWarningResponse]

class YoloAxisRequest(BaseModel):
    image_path: str
    conf: float = 0.25
    kpt_conf: float = 0.25
    model_path: Optional[str] = None

class YoloTicksRequest(BaseModel):
    image_path: str
    conf: float = 0.25
    iou: float = 0.6
    conf_min: float = 0.1
    class_x: int = 0
    class_y: int = 1
    pos_eps: float = 12.0
    ocr_pad: int = 2
    ocr_scale: int = 4
    fix_minus: bool = True
    model_path: Optional[str] = None

@app.post("/evaluate-mask")
async def evaluate_mask(
    image_path: str = Form(...),
    mask_path: str = Form(...)
):
    """
    Принимает пути до исходного изображения и маски (например, из S3),
    очищает график и возвращает путь до сохранённого результата.
    """
    s3 = S3Service()
    service = MaskService(s3)
    result_path = await service.clear_graph(image_path, mask_path)
    return {"clean_path": result_path}

@app.post("/relation")
def get_relation(data: RelationInput):
    relation = PropertyRelation()

    # выбираем метод интерполяции
    if data.method == "cubic":
        relation.fit_cubic(data.T_A, data.A, data.T_B, data.B)
    elif data.method == "linear":
        relation.fit_linear(data.T_A, data.A, data.T_B, data.B)
    else:
        return {"error": f"Метод {data.method} не поддерживается"}

    # получаем таблицу A(B)
    table = relation.get_table()
    if table is None:
        return {"error": "Нет пересечения температур"}

    # преобразуем таблицу в список словарей
    result = table.to_dict(orient="records")
    return {"relation": result}


@app.post("/analysis/fit-model", response_model=FitModelResponse)
def fit_model(payload: dict = Body(...), credentials: HTTPBasicCredentials = Depends(security)):
    authenticate(credentials)

    try:
        data = FitModelRequest(**payload)
    except ValidationError as exc:
        first_error = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(part) for part in first_error.get("loc", []))
        message = first_error.get("msg", "Invalid request body.")
        if location:
            message = f"{location}: {message}"
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "VALIDATION_ERROR", "message": message}},
        )

    try:
        return fit_model_series(data.dict())
    except FitModelError as exc:
        return JSONResponse(status_code=exc.status_code, content=exc.to_response())

@app.post("/spline-interpolate")
def spline_interpolate(data: SplineInterpolationInput):
    try:
        x_new, y_new = PropertyRelation.spline_interpolate(
            x=data.x,
            y=data.y,
            n_points=data.n_points,
            method=data.method,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"x": x_new, "y": y_new}
