from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import OptimizeWarning, curve_fit

VALID_CANDIDATE_MODELS = (
    "linear",
    "poly2",
    "poly3",
    "log",
    "exp",
    "power",
    "spline",
)
VALID_SELECTION_METRICS = ("rmse", "mae", "r2")
VALID_DUPLICATE_X_POLICIES = ("median_y", "mean_y", "first", "error")
DEFAULT_CANDIDATE_MODELS = list(VALID_CANDIDATE_MODELS)
DEFAULT_SELECTION_METRIC = "rmse"
DEFAULT_GRID_SIZE = 200
DEFAULT_ROBUST_CLEANING = True
DEFAULT_DUPLICATE_X_POLICY = "median_y"
DEFAULT_COMPLEXITY_PENALTY_WEIGHT = 0.03
DEFAULT_TIE_THRESHOLD = 0.02
DEFAULT_RETURN_ALL_CANDIDATES = True
MAD_Z_THRESHOLD = 3.5
EPSILON = 1e-12

COMPLEXITY_RANKS = {
    "linear": 1,
    "log": 2,
    "exp": 2,
    "power": 2,
    "poly2": 3,
    "poly3": 4,
    "spline": 5,
}

MODEL_MIN_POINTS = {
    "linear": 2,
    "poly2": 3,
    "poly3": 4,
    "log": 2,
    "exp": 2,
    "power": 2,
    "spline": 4,
}

DOMAIN_ERROR_CODES = {
    "MODEL_DOMAIN_VIOLATION",
    "INSUFFICIENT_POINTS_FOR_MODEL",
}


class FitModelError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = int(status_code)
        self.code = code
        self.message = message

    def to_response(self) -> Dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message}}


class CandidateModelError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class CleanSeriesResult:
    x: np.ndarray
    y: np.ndarray
    report: Dict[str, int]
    warnings: List[Dict[str, str]]


@dataclass
class FittedCandidate:
    name: str
    params: Dict[str, float]
    metrics: Dict[str, float]
    score: float
    complexity_rank: int
    fit_points_used: int
    predictor: Callable[[np.ndarray], np.ndarray]



def fit_model_series(request: Dict[str, Any]) -> Dict[str, Any]:
    normalized = validate_request(request)
    cleaned = clean_series(
        series=normalized["series"],
        robust_cleaning=normalized["robust_cleaning"],
        duplicate_x_policy=normalized["duplicate_x_policy"],
    )

    candidates_response: List[Dict[str, Any]] = []
    valid_candidates: List[FittedCandidate] = []
    warnings_list = list(cleaned.warnings)
    failures: List[CandidateModelError] = []

    for model_name in normalized["candidate_models"]:
        try:
            fitted = fit_candidate(
                name=model_name,
                x=cleaned.x,
                y=cleaned.y,
                selection_metric=normalized["selection_metric"],
                complexity_penalty_weight=normalized["complexity_penalty_weight"],
            )
            valid_candidates.append(fitted)
            candidates_response.append(candidate_to_dict(fitted))
        except CandidateModelError as exc:
            failures.append(exc)
            candidates_response.append(
                {
                    "name": model_name,
                    "status": "invalid",
                    "error_code": exc.code,
                    "error_message": exc.message,
                }
            )
            warnings_list.append(
                {
                    "code": "MODEL_SKIPPED",
                    "message": f"{model_name} skipped: {exc.message}",
                }
            )

    if not valid_candidates:
        if len(normalized["candidate_models"]) == 1 and failures:
            first_failure = failures[0]
            if first_failure.code in DOMAIN_ERROR_CODES:
                raise FitModelError(422, first_failure.code, first_failure.message)
        raise FitModelError(409, "NO_VALID_MODELS", "No valid models could be fitted for the provided series.")

    best_candidate = select_best_candidate(
        candidates=valid_candidates,
        tie_threshold=normalized["tie_threshold"],
    )

    curve_points = build_curve_points(
        predictor=best_candidate.predictor,
        x_min=float(cleaned.x[0]),
        x_max=float(cleaned.x[-1]),
        grid_size=normalized["grid_size"],
    )

    response: Dict[str, Any] = {
        "best_model": candidate_to_dict(best_candidate),
        "candidates": candidates_response if normalized["return_all_candidates"] else [candidate_to_dict(best_candidate)],
        "curve_points": curve_points,
        "cleaning_report": cleaned.report,
        "warnings": warnings_list,
    }
    return response



def validate_request(request: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(request, dict):
        raise FitModelError(422, "VALIDATION_ERROR", "Request body must be an object.")

    series = request.get("series")
    if not isinstance(series, list) or len(series) < 2:
        raise FitModelError(422, "VALIDATION_ERROR", "series must contain at least 2 points.")
    if len(series) > 10000:
        raise FitModelError(422, "VALIDATION_ERROR", "series must contain at most 10000 points.")

    candidate_models = request.get("candidate_models") or list(DEFAULT_CANDIDATE_MODELS)
    if not isinstance(candidate_models, list) or not candidate_models:
        raise FitModelError(422, "VALIDATION_ERROR", "candidate_models must be a non-empty array.")
    normalized_models = [str(model).lower() for model in candidate_models]
    if len(set(normalized_models)) != len(normalized_models):
        raise FitModelError(422, "VALIDATION_ERROR", "candidate_models must be unique.")
    unknown_models = [model for model in normalized_models if model not in VALID_CANDIDATE_MODELS]
    if unknown_models:
        raise FitModelError(422, "VALIDATION_ERROR", f"Unknown candidate models: {', '.join(unknown_models)}")

    selection_metric = str(request.get("selection_metric") or DEFAULT_SELECTION_METRIC).lower()
    if selection_metric not in VALID_SELECTION_METRICS:
        raise FitModelError(422, "VALIDATION_ERROR", "selection_metric must be one of: rmse, mae, r2.")

    grid_size = safe_int(request.get("grid_size", DEFAULT_GRID_SIZE), field_name="grid_size")
    if grid_size < 20 or grid_size > 2000:
        raise FitModelError(422, "VALIDATION_ERROR", "grid_size must be between 20 and 2000.")

    robust_cleaning = bool(request.get("robust_cleaning", DEFAULT_ROBUST_CLEANING))

    duplicate_x_policy = str(request.get("duplicate_x_policy") or DEFAULT_DUPLICATE_X_POLICY).lower()
    if duplicate_x_policy not in VALID_DUPLICATE_X_POLICIES:
        raise FitModelError(
            422,
            "VALIDATION_ERROR",
            "duplicate_x_policy must be one of: median_y, mean_y, first, error.",
        )

    complexity_penalty_weight = safe_float(
        request.get("complexity_penalty_weight", DEFAULT_COMPLEXITY_PENALTY_WEIGHT),
        field_name="complexity_penalty_weight",
    )
    if complexity_penalty_weight < 0 or complexity_penalty_weight > 1:
        raise FitModelError(422, "VALIDATION_ERROR", "complexity_penalty_weight must be between 0 and 1.")

    tie_threshold = safe_float(request.get("tie_threshold", DEFAULT_TIE_THRESHOLD), field_name="tie_threshold")
    if tie_threshold < 0 or tie_threshold > 0.2:
        raise FitModelError(422, "VALIDATION_ERROR", "tie_threshold must be between 0 and 0.2.")

    return_all_candidates = bool(request.get("return_all_candidates", DEFAULT_RETURN_ALL_CANDIDATES))

    return {
        "series": series,
        "candidate_models": normalized_models,
        "selection_metric": selection_metric,
        "grid_size": grid_size,
        "robust_cleaning": robust_cleaning,
        "duplicate_x_policy": duplicate_x_policy,
        "complexity_penalty_weight": complexity_penalty_weight,
        "tie_threshold": tie_threshold,
        "return_all_candidates": return_all_candidates,
    }



def clean_series(
    series: Sequence[Any],
    robust_cleaning: bool = DEFAULT_ROBUST_CLEANING,
    duplicate_x_policy: str = DEFAULT_DUPLICATE_X_POLICY,
) -> CleanSeriesResult:
    input_points = len(series)
    x_values: List[float] = []
    y_values: List[float] = []
    removed_invalid_points = 0

    for point in series:
        x_raw = extract_point_value(point, "x")
        y_raw = extract_point_value(point, "y")

        if x_raw is None or y_raw is None:
            removed_invalid_points += 1
            continue

        try:
            x_float = float(x_raw)
            y_float = float(y_raw)
        except (TypeError, ValueError):
            removed_invalid_points += 1
            continue

        if not (math.isfinite(x_float) and math.isfinite(y_float)):
            removed_invalid_points += 1
            continue

        x_values.append(x_float)
        y_values.append(y_float)

    if len(x_values) < 2:
        raise FitModelError(422, "NO_DATA_AFTER_CLEANING", "Not enough finite data points after cleaning.")

    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    sort_idx = np.argsort(x_arr, kind="mergesort")
    x_arr = x_arr[sort_idx]
    y_arr = y_arr[sort_idx]

    x_arr, y_arr, duplicate_x_groups = dedupe_points(x_arr, y_arr, duplicate_x_policy)

    removed_outliers = 0
    if robust_cleaning and x_arr.size >= 3:
        keep_mask = mad_keep_mask(y_arr)
        removed_outliers = int((~keep_mask).sum())
        x_arr = x_arr[keep_mask]
        y_arr = y_arr[keep_mask]

    unique_x_after_cleaning = int(np.unique(x_arr).size)
    if unique_x_after_cleaning < 2:
        raise FitModelError(422, "NO_DATA_AFTER_CLEANING", "At least 2 unique x values are required after cleaning.")

    warnings_list: List[Dict[str, str]] = []
    if removed_invalid_points:
        warnings_list.append(
            {
                "code": "INVALID_POINTS_REMOVED",
                "message": f"{removed_invalid_points} invalid points removed during cleaning.",
            }
        )
    if duplicate_x_groups:
        warnings_list.append(
            {
                "code": "DUPLICATE_X_AGGREGATED",
                "message": f"{duplicate_x_groups} duplicate x groups aggregated using {duplicate_x_policy}.",
            }
        )
    if removed_outliers:
        warnings_list.append(
            {
                "code": "OUTLIERS_REMOVED",
                "message": f"{removed_outliers} points removed by robust cleaning.",
            }
        )

    report = {
        "input_points": int(input_points),
        "removed_invalid_points": int(removed_invalid_points),
        "removed_outliers": int(removed_outliers),
        "duplicate_x_groups": int(duplicate_x_groups),
        "points_after_cleaning": int(x_arr.size),
        "unique_x_after_cleaning": int(unique_x_after_cleaning),
    }
    return CleanSeriesResult(x=x_arr, y=y_arr, report=report, warnings=warnings_list)



def fit_candidate(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    selection_metric: str,
    complexity_penalty_weight: float,
) -> FittedCandidate:
    validate_model_inputs(name, x, y)

    try:
        params, predictor = build_model(name, x, y)
        predictions = np.asarray(predictor(x), dtype=float)
    except CandidateModelError:
        raise
    except (FloatingPointError, OverflowError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        raise CandidateModelError("MODEL_FIT_FAILED", f"{name} fit failed: {exc}") from exc

    if predictions.shape != y.shape or not np.all(np.isfinite(predictions)):
        raise CandidateModelError("MODEL_PREDICTION_FAILED", f"{name} produced non-finite predictions.")

    metrics = calculate_metrics(y_true=y, y_pred=predictions)
    score = calculate_score(
        metrics=metrics,
        selection_metric=selection_metric,
        complexity_rank=COMPLEXITY_RANKS[name],
        complexity_penalty_weight=complexity_penalty_weight,
    )

    return FittedCandidate(
        name=name,
        params=params,
        metrics=metrics,
        score=score,
        complexity_rank=COMPLEXITY_RANKS[name],
        fit_points_used=int(x.size),
        predictor=predictor,
    )



def validate_model_inputs(name: str, x: np.ndarray, y: np.ndarray) -> None:
    min_points = MODEL_MIN_POINTS[name]
    unique_x = int(np.unique(x).size)
    if unique_x < min_points:
        raise CandidateModelError(
            "INSUFFICIENT_POINTS_FOR_MODEL",
            f"{name} requires at least {min_points} unique x values.",
        )

    if name == "log" and np.any(x <= 0):
        raise CandidateModelError("MODEL_DOMAIN_VIOLATION", "log requires x > 0.")
    if name == "power":
        if np.any(x <= 0):
            raise CandidateModelError("MODEL_DOMAIN_VIOLATION", "power requires x > 0.")
        if np.any(y <= 0):
            raise CandidateModelError("MODEL_DOMAIN_VIOLATION", "power requires y > 0.")



def build_model(name: str, x: np.ndarray, y: np.ndarray) -> tuple[Dict[str, float], Callable[[np.ndarray], np.ndarray]]:
    if name == "linear":
        coeffs = np.polyfit(x, y, 1)
        params = {"k": safe_public_float(coeffs[0]), "b": safe_public_float(coeffs[1])}
        return params, lambda values: np.polyval(coeffs, values)

    if name == "poly2":
        coeffs = np.polyfit(x, y, 2)
        params = {
            "a": safe_public_float(coeffs[0]),
            "b": safe_public_float(coeffs[1]),
            "c": safe_public_float(coeffs[2]),
        }
        return params, lambda values: np.polyval(coeffs, values)

    if name == "poly3":
        coeffs = np.polyfit(x, y, 3)
        params = {
            "a": safe_public_float(coeffs[0]),
            "b": safe_public_float(coeffs[1]),
            "c": safe_public_float(coeffs[2]),
            "d": safe_public_float(coeffs[3]),
        }
        return params, lambda values: np.polyval(coeffs, values)

    if name == "log":
        log_x = np.log(x)
        coeffs = np.polyfit(log_x, y, 1)
        params = {"a": safe_public_float(coeffs[0]), "b": safe_public_float(coeffs[1])}
        return params, lambda values: coeffs[0] * np.log(values) + coeffs[1]

    if name == "exp":
        x_center = float(np.mean(x))
        centered = x - x_center
        initial_a = float(y[0]) if abs(float(y[0])) > EPSILON else 1.0
        initial_b = 0.0

        def exp_model(values: np.ndarray, a: float, b: float) -> np.ndarray:
            exponent = np.clip(b * values, -700, 700)
            return a * np.exp(exponent)

        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            params_raw, _ = curve_fit(
                exp_model,
                centered,
                y,
                p0=(initial_a, initial_b),
                maxfev=20000,
            )

        a_centered = float(params_raw[0])
        b_value = float(params_raw[1])
        a_global = a_centered * math.exp(np.clip(-b_value * x_center, -700, 700))
        params = {"a": safe_public_float(a_global), "b": safe_public_float(b_value)}
        return params, lambda values: exp_model(np.asarray(values, dtype=float) - x_center, a_centered, b_value)

    if name == "power":
        coeffs = np.polyfit(np.log(x), np.log(y), 1)
        b_value = float(coeffs[0])
        a_value = math.exp(float(coeffs[1]))
        params = {"a": safe_public_float(a_value), "b": safe_public_float(b_value)}
        return params, lambda values: a_value * np.power(values, b_value)

    if name == "spline":
        spline = CubicSpline(x, y, extrapolate=False)
        return {"kind": "cubic"}, lambda values: spline(values)

    raise CandidateModelError("VALIDATION_ERROR", f"Unsupported model '{name}'.")



def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    residuals = np.asarray(y_true - y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    mae = float(np.mean(np.abs(residuals)))

    ss_res = float(np.sum(np.square(residuals)))
    ss_tot = float(np.sum(np.square(y_true - np.mean(y_true))))
    if ss_tot <= EPSILON:
        r2 = 1.0 if ss_res <= EPSILON else 0.0
    else:
        r2 = 1.0 - (ss_res / ss_tot)

    return {
        "rmse": safe_public_float(rmse),
        "mae": safe_public_float(mae),
        "r2": safe_public_float(r2),
    }



def calculate_score(
    metrics: Dict[str, float],
    selection_metric: str,
    complexity_rank: int,
    complexity_penalty_weight: float,
) -> float:
    if selection_metric == "rmse":
        base_score = metrics["rmse"]
    elif selection_metric == "mae":
        base_score = metrics["mae"]
    else:
        base_score = 1.0 - metrics["r2"]

    score = float(base_score + (complexity_penalty_weight * complexity_rank))
    return safe_public_float(score)



def select_best_candidate(candidates: Sequence[FittedCandidate], tie_threshold: float) -> FittedCandidate:
    ranked = sorted(candidates, key=lambda candidate: candidate.score)
    best = ranked[0]

    for candidate in ranked[1:]:
        if abs(candidate.score - best.score) > tie_threshold:
            break
        if candidate.complexity_rank < best.complexity_rank:
            best = candidate
            continue
        if candidate.complexity_rank == best.complexity_rank:
            if candidate.metrics["rmse"] < best.metrics["rmse"] - EPSILON:
                best = candidate
                continue
            if abs(candidate.metrics["rmse"] - best.metrics["rmse"]) <= EPSILON and candidate.name < best.name:
                best = candidate

    return best



def build_curve_points(
    predictor: Callable[[np.ndarray], np.ndarray],
    x_min: float,
    x_max: float,
    grid_size: int,
) -> List[Dict[str, float]]:
    x_grid = np.linspace(float(x_min), float(x_max), int(grid_size))
    y_grid = np.asarray(predictor(x_grid), dtype=float)
    if y_grid.shape != x_grid.shape or not np.all(np.isfinite(y_grid)):
        raise FitModelError(422, "BEST_MODEL_PREDICTION_FAILED", "Best model produced non-finite curve points.")

    return [
        {"x": safe_public_float(x_value), "y": safe_public_float(y_value)}
        for x_value, y_value in zip(x_grid.tolist(), y_grid.tolist())
    ]



def dedupe_points(x: np.ndarray, y: np.ndarray, duplicate_x_policy: str) -> tuple[np.ndarray, np.ndarray, int]:
    deduped_x: List[float] = []
    deduped_y: List[float] = []
    duplicate_groups = 0
    start = 0

    while start < x.size:
        end = start + 1
        while end < x.size and x[end] == x[start]:
            end += 1

        group_y = y[start:end]
        if end - start > 1:
            duplicate_groups += 1
            if duplicate_x_policy == "error":
                raise FitModelError(422, "DUPLICATE_X_NOT_ALLOWED", "Duplicate x values are not allowed.")
            if duplicate_x_policy == "median_y":
                y_value = float(np.median(group_y))
            elif duplicate_x_policy == "mean_y":
                y_value = float(np.mean(group_y))
            else:
                y_value = float(group_y[0])
        else:
            y_value = float(group_y[0])

        deduped_x.append(float(x[start]))
        deduped_y.append(y_value)
        start = end

    return np.asarray(deduped_x, dtype=float), np.asarray(deduped_y, dtype=float), duplicate_groups



def mad_keep_mask(y: np.ndarray) -> np.ndarray:
    median = float(np.median(y))
    abs_deviation = np.abs(y - median)
    mad = float(np.median(abs_deviation))
    if mad <= EPSILON:
        return np.ones_like(y, dtype=bool)
    modified_z = 0.6745 * (y - median) / mad
    return np.abs(modified_z) <= MAD_Z_THRESHOLD



def candidate_to_dict(candidate: FittedCandidate) -> Dict[str, Any]:
    return {
        "name": candidate.name,
        "status": "valid",
        "params": candidate.params,
        "metrics": candidate.metrics,
        "score": safe_public_float(candidate.score),
        "complexity_rank": int(candidate.complexity_rank),
        "fit_points_used": int(candidate.fit_points_used),
    }



def safe_float(value: Any, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FitModelError(422, "VALIDATION_ERROR", f"{field_name} must be numeric.") from exc

    if not math.isfinite(result):
        raise FitModelError(422, "VALIDATION_ERROR", f"{field_name} must be finite.")
    return result



def safe_int(value: Any, field_name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise FitModelError(422, "VALIDATION_ERROR", f"{field_name} must be an integer.") from exc
    return result



def safe_public_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise CandidateModelError("MODEL_FIT_FAILED", "Model produced a non-finite numeric value.")
    return result



def extract_point_value(point: Any, field_name: str) -> Any:
    if isinstance(point, dict):
        return point.get(field_name)
    return getattr(point, field_name, None)
