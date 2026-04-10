from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import UnivariateSpline
from scipy.signal import argrelextrema

__all__ = ["build_curve_features_response", "CurveFeaturesError"]

VALID_BASE_MODELS = ("spline", "poly2", "poly3")
MIN_POINTS_FOR_MODEL = {
    "spline": 5,
    "poly2": 3,
    "poly3": 4,
}

DEFAULT_BASE_MODEL = "spline"
DEFAULT_GRID_SIZE = 300
MIN_GRID_SIZE = 100
MAX_GRID_SIZE = 1000
DEFAULT_ROBUST_CLEANING = False
DEFAULT_RETURN_DEBUG = False
DEFAULT_EXTREMA_MIN_DELTA = 0.01
DEFAULT_DERIVATIVE_ZERO_EPSILON = 1e-6
MAX_SERIES_POINTS = 10000
MAD_OUTLIER_THRESHOLD = 5.0
EPSILON = 1e-12


class CurveFeaturesError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = int(status_code)
        self.code = code
        self.message = message

    def to_response(self) -> Dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message}}


@dataclass
class PreparedSeries:
    x: np.ndarray
    y: np.ndarray
    cleaned_series: List[Dict[str, float]]
    warnings: List[Dict[str, Any]]
    report: Dict[str, Any]


@dataclass
class BaseCurveModel:
    predictor: Callable[[np.ndarray], np.ndarray]
    debug: Dict[str, Any]


def build_curve_features_response(payload: dict) -> Dict[str, Any]:
    normalized = _validate_payload(payload)
    prepared = _prepare_series(
        series=normalized["series"],
        base_model=normalized["base_model"],
        robust_cleaning=normalized["robust_cleaning"],
        x_range=normalized["x_range"],
    )

    model = _build_base_model(
        x=prepared.x,
        y=prepared.y,
        base_model=normalized["base_model"],
        smoothing_factor=normalized["smoothing_factor"],
    )

    x_grid = np.linspace(float(prepared.x[0]), float(prepared.x[-1]), normalized["grid_size"])
    try:
        y_grid = np.asarray(model.predictor(x_grid), dtype=float)
    except CurveFeaturesError:
        raise
    except Exception as exc:
        raise CurveFeaturesError(500, "MODEL_EVALUATION_FAILED", f"Failed to evaluate base model: {exc}") from exc

    if y_grid.shape != x_grid.shape or not np.all(np.isfinite(y_grid)):
        raise CurveFeaturesError(500, "MODEL_EVALUATION_FAILED", "Base model produced non-finite values on the analysis grid.")

    dy_dx = np.gradient(y_grid, x_grid)
    d2y_dx2 = np.gradient(dy_dx, x_grid)

    y_span = float(np.max(y_grid) - np.min(y_grid))
    extrema_threshold = max(normalized["extrema_min_delta"], normalized["extrema_min_delta"] * max(y_span, 1.0))

    extrema = _detect_extrema(x_grid=x_grid, y_grid=y_grid, min_delta=extrema_threshold)
    inflection_points = _detect_inflections(x_grid=x_grid, y_grid=y_grid, d2y_dx2=d2y_dx2)
    monotonic_segments = _build_monotonic_segments(
        x_grid=x_grid,
        dy_dx=dy_dx,
        zero_epsilon=normalized["derivative_zero_epsilon"],
    )

    integral_value = float(simpson(y_grid, x=x_grid))
    warnings = list(normalized["warnings"])
    warnings.extend(prepared.warnings)
    if model.debug.get("increased_smoothing"):
        warnings.append(
            {
                "code": "SPLINE_SMOOTHING_INCREASED",
                "message": "Spline fitting required increased smoothing to obtain a stable base curve.",
            }
        )
    warnings.extend(_build_analysis_warnings(prepared=prepared, x_grid=x_grid, extrema=extrema))

    debug: Optional[Dict[str, Any]] = None
    if normalized["return_debug"]:
        debug = {
            "input_point_count": prepared.report["input_points"],
            "cleaned_point_count": int(prepared.x.size),
            "grid_size_requested": normalized["grid_size_requested"],
            "grid_size_used": normalized["grid_size"],
            "base_model": normalized["base_model"],
            "x_domain": {
                "min": float(prepared.x[0]),
                "max": float(prepared.x[-1]),
            },
            "cleaning_report": prepared.report,
            "model": model.debug,
            "extrema_min_delta_used": float(extrema_threshold),
            "derivative_zero_epsilon": float(normalized["derivative_zero_epsilon"]),
        }

    return {
        "cleaned_series": prepared.cleaned_series,
        "base_curve": _to_curve_points(x_grid, y_grid),
        "first_derivative_curve": _to_curve_points(x_grid, dy_dx),
        "second_derivative_curve": _to_curve_points(x_grid, d2y_dx2),
        "extrema": extrema,
        "inflection_points": inflection_points,
        "monotonic_segments": monotonic_segments,
        "integral": {
            "value": integral_value,
            "x_min": float(x_grid[0]),
            "x_max": float(x_grid[-1]),
        },
        "warnings": warnings,
        "debug": debug,
    }


def _validate_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise CurveFeaturesError(422, "VALIDATION_ERROR", "Request body must be an object.")

    series = payload.get("series")
    if not isinstance(series, list) or len(series) < 2:
        raise CurveFeaturesError(422, "VALIDATION_ERROR", "series must contain at least 2 points.")
    if len(series) > MAX_SERIES_POINTS:
        raise CurveFeaturesError(422, "VALIDATION_ERROR", f"series must contain at most {MAX_SERIES_POINTS} points.")

    base_model = str(payload.get("base_model") or DEFAULT_BASE_MODEL).lower()
    if base_model not in VALID_BASE_MODELS:
        raise CurveFeaturesError(422, "VALIDATION_ERROR", "base_model must be one of: spline, poly2, poly3.")

    grid_size_requested = _safe_int(payload.get("grid_size", DEFAULT_GRID_SIZE), field_name="grid_size")
    grid_size = int(max(MIN_GRID_SIZE, min(MAX_GRID_SIZE, grid_size_requested)))
    warnings: List[Dict[str, Any]] = []
    if grid_size != grid_size_requested:
        warnings.append(
            {
                "code": "GRID_SIZE_CLAMPED",
                "message": f"grid_size was clamped from {grid_size_requested} to {grid_size}.",
            }
        )

    smoothing_raw = payload.get("smoothing_factor")
    smoothing_factor = None
    if smoothing_raw is not None:
        smoothing_factor = _safe_float(smoothing_raw, field_name="smoothing_factor")
        if smoothing_factor < 0:
            raise CurveFeaturesError(422, "VALIDATION_ERROR", "smoothing_factor must be greater than or equal to 0.")

    x_range = _validate_x_range(payload.get("x_range"))

    extrema_min_delta = _safe_float(
        payload.get("extrema_min_delta", DEFAULT_EXTREMA_MIN_DELTA),
        field_name="extrema_min_delta",
    )
    if extrema_min_delta < 0:
        raise CurveFeaturesError(422, "VALIDATION_ERROR", "extrema_min_delta must be greater than or equal to 0.")

    derivative_zero_epsilon = _safe_float(
        payload.get("derivative_zero_epsilon", DEFAULT_DERIVATIVE_ZERO_EPSILON),
        field_name="derivative_zero_epsilon",
    )
    if derivative_zero_epsilon <= 0:
        raise CurveFeaturesError(422, "VALIDATION_ERROR", "derivative_zero_epsilon must be greater than 0.")

    return {
        "series": series,
        "base_model": base_model,
        "grid_size_requested": grid_size_requested,
        "grid_size": grid_size,
        "smoothing_factor": smoothing_factor,
        "robust_cleaning": bool(payload.get("robust_cleaning", DEFAULT_ROBUST_CLEANING)),
        "x_range": x_range,
        "return_debug": bool(payload.get("return_debug", DEFAULT_RETURN_DEBUG)),
        "extrema_min_delta": extrema_min_delta,
        "derivative_zero_epsilon": derivative_zero_epsilon,
        "warnings": warnings,
    }


def _validate_x_range(raw_range: Any) -> Optional[Dict[str, float]]:
    if raw_range is None:
        return None
    if not isinstance(raw_range, dict):
        raise CurveFeaturesError(422, "VALIDATION_ERROR", "x_range must be an object with min and max.")

    if "min" not in raw_range or "max" not in raw_range:
        raise CurveFeaturesError(422, "VALIDATION_ERROR", "x_range must contain min and max.")

    x_min = _safe_float(raw_range.get("min"), field_name="x_range.min")
    x_max = _safe_float(raw_range.get("max"), field_name="x_range.max")
    if x_min >= x_max:
        raise CurveFeaturesError(422, "VALIDATION_ERROR", "x_range.min must be less than x_range.max.")
    return {"min": x_min, "max": x_max}


def _prepare_series(
    series: Sequence[Any],
    base_model: str,
    robust_cleaning: bool,
    x_range: Optional[Dict[str, float]],
) -> PreparedSeries:
    warnings: List[Dict[str, Any]] = []
    x_values: List[float] = []
    y_values: List[float] = []
    removed_invalid_points = 0

    for point in series:
        x_raw = _extract_point_value(point, "x")
        y_raw = _extract_point_value(point, "y")
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

    if removed_invalid_points:
        warnings.append(
            {
                "code": "INVALID_POINTS_REMOVED",
                "message": f"{removed_invalid_points} invalid points were removed during cleaning.",
            }
        )

    if len(x_values) < 2:
        raise CurveFeaturesError(422, "EMPTY_SERIES_AFTER_CLEANING", "No valid points remain after cleaning.")

    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)

    sort_idx = np.argsort(x_arr, kind="mergesort")
    x_sorted = x_arr[sort_idx]
    y_sorted = y_arr[sort_idx]

    unique_x: List[float] = []
    aggregated_y: List[float] = []
    duplicate_points_aggregated = 0
    duplicate_groups = 0
    idx = 0
    while idx < x_sorted.size:
        current_x = x_sorted[idx]
        end = idx + 1
        while end < x_sorted.size and math.isclose(x_sorted[end], current_x, rel_tol=0.0, abs_tol=EPSILON):
            end += 1

        y_group = y_sorted[idx:end]
        unique_x.append(float(current_x))
        aggregated_y.append(float(np.median(y_group)))
        if end - idx > 1:
            duplicate_points_aggregated += (end - idx) - 1
            duplicate_groups += 1
        idx = end

    if duplicate_points_aggregated:
        warnings.append(
            {
                "code": "DUPLICATE_X_AGGREGATED",
                "message": "Duplicate x values were aggregated using median(y).",
                "count": int(duplicate_points_aggregated),
                "groups": int(duplicate_groups),
            }
        )

    x_unique = np.asarray(unique_x, dtype=float)
    y_unique = np.asarray(aggregated_y, dtype=float)

    points_removed_by_range = 0
    range_fraction = 1.0
    if x_range is not None:
        original_span = float(x_unique[-1] - x_unique[0]) if x_unique.size > 1 else 0.0
        range_mask = (x_unique >= x_range["min"]) & (x_unique <= x_range["max"])
        points_removed_by_range = int(np.count_nonzero(~range_mask))
        x_unique = x_unique[range_mask]
        y_unique = y_unique[range_mask]

        if x_unique.size == 0:
            raise CurveFeaturesError(422, "X_RANGE_EMPTY", "No points remain inside the requested x_range.")

        filtered_span = float(x_unique[-1] - x_unique[0]) if x_unique.size > 1 else 0.0
        if original_span > EPSILON:
            range_fraction = filtered_span / original_span

        if points_removed_by_range:
            warnings.append(
                {
                    "code": "X_RANGE_APPLIED",
                    "message": f"x_range filter removed {points_removed_by_range} points outside the requested domain.",
                }
            )

    outlier_count = _count_outliers(y_unique)
    removed_outliers = 0
    if outlier_count:
        if robust_cleaning:
            mask = ~_mad_outlier_mask(y_unique)
            removed_outliers = int(np.count_nonzero(~mask))
            x_unique = x_unique[mask]
            y_unique = y_unique[mask]
            warnings.append(
                {
                    "code": "ROBUST_OUTLIERS_REMOVED",
                    "message": f"Robust cleaning removed {removed_outliers} strong y outliers using MAD.",
                }
            )
        else:
            warnings.append(
                {
                    "code": "OUTLIERS_DETECTED",
                    "message": f"{outlier_count} strong y outliers were detected but not removed.",
                }
            )

    min_points = MIN_POINTS_FOR_MODEL[base_model]
    if x_unique.size < min_points:
        raise CurveFeaturesError(
            422,
            "NOT_ENOUGH_POINTS",
            f"At least {min_points} unique points are required for {base_model}-based derivative analysis.",
        )

    x_span = float(x_unique[-1] - x_unique[0])
    if x_span <= EPSILON:
        raise CurveFeaturesError(422, "CONSTANT_X", "All remaining x values are identical after cleaning.")

    if x_unique.size <= min_points + 1:
        warnings.append(
            {
                "code": "LOW_POINT_COUNT",
                "message": f"Derivative analysis may be unstable because only {int(x_unique.size)} unique points remain after cleaning.",
            }
        )

    if range_fraction < 0.1:
        warnings.append(
            {
                "code": "NARROW_X_RANGE",
                "message": "The selected x_range is narrow relative to the original domain; derivatives may be unstable.",
            }
        )

    spacing = np.diff(x_unique)
    if spacing.size and float(np.max(spacing) / max(np.min(spacing), EPSILON)) > 20:
        warnings.append(
            {
                "code": "IRREGULAR_X_SPACING",
                "message": "Input points are unevenly spaced; local derivative features may be sensitive to interpolation.",
            }
        )

    cleaned_series = [
        {"x": float(x_value), "y": float(y_value)}
        for x_value, y_value in zip(x_unique, y_unique)
    ]

    return PreparedSeries(
        x=x_unique,
        y=y_unique,
        cleaned_series=cleaned_series,
        warnings=warnings,
        report={
            "input_points": int(len(series)),
            "removed_invalid_points": int(removed_invalid_points),
            "duplicate_points_aggregated": int(duplicate_points_aggregated),
            "duplicate_groups": int(duplicate_groups),
            "points_removed_by_range": int(points_removed_by_range),
            "outliers_detected": int(outlier_count),
            "outliers_removed": int(removed_outliers),
        },
    )


def _build_base_model(
    x: np.ndarray,
    y: np.ndarray,
    base_model: str,
    smoothing_factor: Optional[float],
) -> BaseCurveModel:
    if base_model == "spline":
        requested_s = smoothing_factor
        used_s = float(len(x)) if smoothing_factor is None else float(smoothing_factor)
        warning_needed = False
        try:
            spline = UnivariateSpline(x, y, s=used_s)
        except Exception as exc:
            if smoothing_factor is not None:
                raise CurveFeaturesError(500, "SPLINE_BUILD_FAILED", f"Failed to build spline model: {exc}") from exc
            used_s = float(len(x)) * 2.0
            warning_needed = True
            try:
                spline = UnivariateSpline(x, y, s=used_s)
            except Exception as retry_exc:
                raise CurveFeaturesError(500, "SPLINE_BUILD_FAILED", f"Failed to build spline model: {retry_exc}") from retry_exc

        return BaseCurveModel(
            predictor=lambda x_new: np.asarray(spline(x_new), dtype=float),
            debug={
                "type": "spline",
                "smoothing_factor_requested": requested_s,
                "smoothing_factor_used": used_s,
                "residual": float(spline.get_residual()),
                "increased_smoothing": warning_needed,
            },
        )

    degree = 2 if base_model == "poly2" else 3
    try:
        coeffs = np.polyfit(x, y, deg=degree)
    except Exception as exc:
        raise CurveFeaturesError(500, "POLYFIT_FAILED", f"Failed to build polynomial model: {exc}") from exc

    poly = np.poly1d(coeffs)
    return BaseCurveModel(
        predictor=lambda x_new: np.asarray(poly(x_new), dtype=float),
        debug={
            "type": base_model,
            "degree": degree,
            "coefficients": [float(value) for value in coeffs.tolist()],
        },
    )


def _detect_extrema(x_grid: np.ndarray, y_grid: np.ndarray, min_delta: float) -> List[Dict[str, Any]]:
    maxima_idx = argrelextrema(y_grid, np.greater)[0]
    minima_idx = argrelextrema(y_grid, np.less)[0]

    extrema: List[Dict[str, Any]] = []
    for indices, kind in ((maxima_idx, "max"), (minima_idx, "min")):
        for idx in indices.tolist():
            if idx <= 0 or idx >= len(y_grid) - 1:
                continue
            left_delta = abs(float(y_grid[idx] - y_grid[idx - 1]))
            right_delta = abs(float(y_grid[idx] - y_grid[idx + 1]))
            if min(left_delta, right_delta) < min_delta:
                continue
            extrema.append(
                {
                    "x": float(x_grid[idx]),
                    "y": float(y_grid[idx]),
                    "type": kind,
                    "source_index": int(idx),
                }
            )

    extrema.sort(key=lambda item: item["source_index"])
    return extrema


def _detect_inflections(x_grid: np.ndarray, y_grid: np.ndarray, d2y_dx2: np.ndarray) -> List[Dict[str, Any]]:
    signs = np.sign(d2y_dx2).astype(int)
    signs = _fill_zero_signs(signs)
    change_idx = np.where(np.diff(signs) != 0)[0]

    return [
        {
            "x": float(x_grid[idx]),
            "y": float(y_grid[idx]),
            "source_index": int(idx),
        }
        for idx in change_idx.tolist()
    ]


def _fill_zero_signs(signs: np.ndarray) -> np.ndarray:
    if signs.size == 0:
        return signs

    filled = signs.copy()
    for idx in range(1, filled.size):
        if filled[idx] == 0:
            filled[idx] = filled[idx - 1]
    for idx in range(filled.size - 2, -1, -1):
        if filled[idx] == 0:
            filled[idx] = filled[idx + 1]
    return filled


def _build_monotonic_segments(
    x_grid: np.ndarray,
    dy_dx: np.ndarray,
    zero_epsilon: float,
) -> List[Dict[str, Any]]:
    if x_grid.size == 0:
        return []

    trends = [_trend_from_derivative(value, zero_epsilon) for value in dy_dx]
    segments: List[Dict[str, Any]] = []
    start_idx = 0
    current_trend = trends[0]

    for idx in range(1, len(trends)):
        if trends[idx] == current_trend:
            continue
        segments.append(
            {
                "from_x": float(x_grid[start_idx]),
                "to_x": float(x_grid[idx - 1]),
                "trend": current_trend,
            }
        )
        start_idx = idx
        current_trend = trends[idx]

    segments.append(
        {
            "from_x": float(x_grid[start_idx]),
            "to_x": float(x_grid[-1]),
            "trend": current_trend,
        }
    )
    return segments


def _build_analysis_warnings(
    prepared: PreparedSeries,
    x_grid: np.ndarray,
    extrema: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    warnings: List[Dict[str, Any]] = []

    if len(extrema) > max(8, int(len(x_grid) * 0.05)):
        warnings.append(
            {
                "code": "NOISY_EXTREMA_PATTERN",
                "message": "A large number of extrema was detected; the curve may still contain noise-driven oscillations.",
            }
        )

    model_debug = prepared.report
    if model_debug["outliers_removed"] > 0 and model_debug["outliers_removed"] >= max(3, model_debug["input_points"] // 4):
        warnings.append(
            {
                "code": "AGGRESSIVE_CLEANING",
                "message": "Robust cleaning removed a substantial fraction of points; review the cleaned series before using derivatives.",
            }
        )

    return warnings


def _to_curve_points(x_values: np.ndarray, y_values: np.ndarray) -> List[Dict[str, float]]:
    return [
        {"x": float(x_value), "y": float(y_value)}
        for x_value, y_value in zip(x_values, y_values)
    ]


def _mad_outlier_mask(values: np.ndarray) -> np.ndarray:
    if values.size < 3:
        return np.zeros(values.shape, dtype=bool)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad <= EPSILON:
        return np.zeros(values.shape, dtype=bool)
    modified_z = 0.6745 * (values - median) / mad
    return np.abs(modified_z) > MAD_OUTLIER_THRESHOLD


def _count_outliers(values: np.ndarray) -> int:
    return int(np.count_nonzero(_mad_outlier_mask(values)))


def _trend_from_derivative(value: float, zero_epsilon: float) -> str:
    if abs(float(value)) < zero_epsilon:
        return "flat"
    if value > 0:
        return "increasing"
    return "decreasing"


def _extract_point_value(point: Any, key: str) -> Any:
    if isinstance(point, dict):
        return point.get(key)
    return getattr(point, key, None)


def _safe_int(value: Any, field_name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise CurveFeaturesError(422, "VALIDATION_ERROR", f"{field_name} must be an integer.")
    return result


def _safe_float(value: Any, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise CurveFeaturesError(422, "VALIDATION_ERROR", f"{field_name} must be a number.")
    if not math.isfinite(result):
        raise CurveFeaturesError(422, "VALIDATION_ERROR", f"{field_name} must be finite.")
    return result
