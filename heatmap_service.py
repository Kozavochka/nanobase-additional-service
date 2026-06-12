from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.interpolate import griddata

MIN_POINTS_REQUIRED = 10
LOW_POINTS_WARNING_THRESHOLD = 15
MAX_NAN_RATIO_WARNING = 0.5


class HeatmapError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = int(status_code)
        self.code = code
        self.message = message

    def to_response(self) -> Dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message}}


def build_heatmap_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    points_raw = payload.get("points") or []
    method = str(payload.get("method") or "griddata").lower()
    if method != "griddata":
        raise HeatmapError(422, "UNSUPPORTED_METHOD", "Only griddata method is supported.")

    interpolation_requested = str(payload.get("interpolation") or "linear").lower()
    if interpolation_requested not in {"nearest", "linear", "cubic"}:
        raise HeatmapError(422, "UNSUPPORTED_INTERPOLATION", "Interpolation must be nearest, linear, or cubic.")

    grid_size_x = int(payload.get("grid_size_x") or 100)
    grid_size_y = int(payload.get("grid_size_y") or 100)
    if grid_size_x < 30 or grid_size_x > 300:
        raise HeatmapError(422, "INVALID_GRID_SIZE", "grid_size_x must be between 30 and 300.")
    if grid_size_y < 30 or grid_size_y > 300:
        raise HeatmapError(422, "INVALID_GRID_SIZE", "grid_size_y must be between 30 and 300.")
    show_original_points = bool(payload.get("show_original_points", True))
    return_debug = bool(payload.get("return_debug", False))

    x1_range = payload.get("x1_range")
    x2_range = payload.get("x2_range")

    cleaned_points = _clean_points(points_raw, x1_range=x1_range, x2_range=x2_range)
    if len(cleaned_points) < MIN_POINTS_REQUIRED:
        cleaned_count = len(cleaned_points)
        raise HeatmapError(
            422,
            "INSUFFICIENT_POINTS",
            f"At least {MIN_POINTS_REQUIRED} valid points are required to build heatmap. "
            f"Got {cleaned_count} after filtering/validation.",
        )

    aggregated_points, duplicate_group_count = _aggregate_duplicate_coordinates(cleaned_points)
    if len(aggregated_points) < MIN_POINTS_REQUIRED:
        aggregated_count = len(aggregated_points)
        raise HeatmapError(
            422,
            "INSUFFICIENT_POINTS",
            f"At least {MIN_POINTS_REQUIRED} points are required after duplicate aggregation. "
            f"Got {aggregated_count} unique coordinate points.",
        )

    x1_values = np.asarray([point["x1"] for point in aggregated_points], dtype=float)
    x2_values = np.asarray([point["x2"] for point in aggregated_points], dtype=float)
    z_values = np.asarray([point["z"] for point in aggregated_points], dtype=float)

    if np.unique(x1_values).size < 2:
        raise HeatmapError(422, "X1_VARIATION_REQUIRED", "Heatmap requires at least two distinct x1 values.")
    if np.unique(x2_values).size < 2:
        raise HeatmapError(422, "X2_VARIATION_REQUIRED", "Heatmap requires at least two distinct x2 values.")

    warnings: List[Dict[str, str | int]] = []
    if len(aggregated_points) < LOW_POINTS_WARNING_THRESHOLD:
        warnings.append(
            {
                "code": "LOW_POINT_COUNT",
                "message": "Point count is low; interpolation may be unstable.",
            }
        )

    if duplicate_group_count > 0:
        warnings.append(
            {
                "code": "DUPLICATE_COORDINATES_AGGREGATED",
                "message": "Points with identical x1/x2 were aggregated using median(z).",
                "count": int(duplicate_group_count),
            }
        )

    interpolation_used = interpolation_requested
    if interpolation_requested == "cubic" and len(aggregated_points) < 20:
        interpolation_used = "linear"
        warnings.append(
            {
                "code": "CUBIC_FALLBACK_TO_LINEAR",
                "message": "Cubic interpolation replaced by linear due to sparse data.",
            }
        )

    x1_min = float(np.min(x1_values))
    x1_max = float(np.max(x1_values))
    x2_min = float(np.min(x2_values))
    x2_max = float(np.max(x2_values))

    x1_grid = np.linspace(x1_min, x1_max, grid_size_x)
    x2_grid = np.linspace(x2_min, x2_max, grid_size_y)
    mesh_x1, mesh_x2 = np.meshgrid(x1_grid, x2_grid)

    z_mesh = griddata(
        points=(x1_values, x2_values),
        values=z_values,
        xi=(mesh_x1, mesh_x2),
        method=interpolation_used,
    )

    if z_mesh is None:
        raise HeatmapError(422, "INTERPOLATION_FAILED", "Interpolation returned no result.")

    valid_mask = ~np.isnan(z_mesh)
    nan_ratio = 1.0 - float(np.mean(valid_mask))
    if nan_ratio > MAX_NAN_RATIO_WARNING:
        warnings.append(
            {
                "code": "HIGH_EMPTY_AREA_RATIO",
                "message": "Large part of the grid is not interpolated (NaN cells).",
            }
        )

    z_grid: List[List[float | None]] = []
    for row in z_mesh.tolist():
        converted_row: List[float | None] = []
        for value in row:
            if value is None or not math.isfinite(float(value)):
                converted_row.append(None)
            else:
                converted_row.append(float(value))
        z_grid.append(converted_row)

    response: Dict[str, Any] = {
        "x1_grid": [float(value) for value in x1_grid.tolist()],
        "x2_grid": [float(value) for value in x2_grid.tolist()],
        "z_grid": z_grid,
        "valid_mask": valid_mask.tolist(),
        "original_points": aggregated_points if show_original_points else [],
        "summary": {
            "point_count_raw": int(len(points_raw)),
            "point_count_cleaned": int(len(aggregated_points)),
            "x1_min": x1_min,
            "x1_max": x1_max,
            "x2_min": x2_min,
            "x2_max": x2_max,
            "z_min": float(np.min(z_values)),
            "z_max": float(np.max(z_values)),
            "nan_ratio": float(nan_ratio),
        },
        "warnings": warnings,
    }

    if return_debug:
        response["debug"] = {
            "interpolation_requested": interpolation_requested,
            "interpolation_used": interpolation_used,
            "duplicate_group_count": int(duplicate_group_count),
        }

    return response


def _clean_points(
    points: Sequence[Any],
    x1_range: Dict[str, Any] | None,
    x2_range: Dict[str, Any] | None,
) -> List[Dict[str, float]]:
    result: List[Dict[str, float]] = []
    x1_min, x1_max = _parse_range(x1_range, "x1")
    x2_min, x2_max = _parse_range(x2_range, "x2")

    for point in points:
        if not isinstance(point, dict):
            continue
        x1 = _safe_float(point.get("x1"))
        x2 = _safe_float(point.get("x2"))
        z = _safe_float(point.get("z"))
        if x1 is None or x2 is None or z is None:
            continue

        if x1_min is not None and x1 < x1_min:
            continue
        if x1_max is not None and x1 > x1_max:
            continue
        if x2_min is not None and x2 < x2_min:
            continue
        if x2_max is not None and x2 > x2_max:
            continue

        result.append({"x1": x1, "x2": x2, "z": z})

    return result


def _parse_range(range_payload: Dict[str, Any] | None, axis_name: str) -> Tuple[float | None, float | None]:
    if not range_payload:
        return None, None

    min_value = _safe_float(range_payload.get("min"))
    max_value = _safe_float(range_payload.get("max"))
    if min_value is None or max_value is None:
        raise HeatmapError(422, "INVALID_RANGE", f"{axis_name}_range.min and max must be finite numbers.")
    if min_value >= max_value:
        raise HeatmapError(422, "INVALID_RANGE", f"{axis_name}_range.min must be less than max.")

    return min_value, max_value


def _aggregate_duplicate_coordinates(points: Sequence[Dict[str, float]]) -> Tuple[List[Dict[str, float]], int]:
    grouped: Dict[Tuple[float, float], List[float]] = {}
    for point in points:
        key = (point["x1"], point["x2"])
        grouped.setdefault(key, []).append(point["z"])

    aggregated: List[Dict[str, float]] = []
    duplicate_group_count = 0
    for (x1, x2), z_values in grouped.items():
        if len(z_values) > 1:
            duplicate_group_count += 1
        aggregated.append({"x1": float(x1), "x2": float(x2), "z": float(np.median(np.asarray(z_values, dtype=float)))})

    return aggregated, duplicate_group_count


def _safe_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(converted):
        return None
    return converted
