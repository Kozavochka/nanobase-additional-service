from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal, Dict, Any
import math
import numpy as np
import pandas as pd

ScaleType = Literal["linear", "log10"]

@dataclass
class AxisConfig:
    scale: ScaleType = "linear"
    vmin: float = 0.0
    vmax: float = 1.0
    ticks: Optional[List[float]] = None
    tick_tolerance: Optional[float] = None

@dataclass
class ROI:
    x_left: float
    x_right: float
    y_top: float
    y_bottom: float

class PlotCalibrator:
    def __init__(self, roi: ROI, xaxis: AxisConfig, yaxis: AxisConfig):
        assert roi.x_right > roi.x_left and roi.y_bottom > roi.y_top, "Некорректный ROI"
        assert xaxis.vmax != xaxis.vmin and yaxis.vmax != yaxis.vmin, "Диапазон осей не может быть нулевым"
        self.roi = roi
        self.xaxis = xaxis
        self.yaxis = yaxis

    def _map_linear(self, t: float, t0: float, t1: float, v0: float, v1: float) -> float:
        return v0 + (t - t0) / (t1 - t0) * (v1 - v0)

    def _map_log10(self, t: float, t0: float, t1: float, v0: float, v1: float) -> float:
        log0, log1 = math.log10(v0), math.log10(v1)
        logv = self._map_linear(t, t0, t1, log0, log1)
        return 10 ** logv

    def _snap(self, value: float, ticks: Optional[List[float]], tol: Optional[float]) -> float:
        if not ticks or tol is None:
            return value
        idx = int(np.argmin([abs(value - tv) for tv in ticks]))
        nearest = ticks[idx]
        return nearest if abs(value - nearest) <= tol else value

    def pixel_to_values(self, x_pix: float, y_pix: float) -> Tuple[float, float]:
        if self.xaxis.scale == "linear":
            x_val = self._map_linear(x_pix, self.roi.x_left, self.roi.x_right, self.xaxis.vmin, self.xaxis.vmax)
        else:
            x_val = self._map_log10(x_pix, self.roi.x_left, self.roi.x_right, self.xaxis.vmin, self.xaxis.vmax)

        if self.yaxis.scale == "linear":
            y_val = self._map_linear(y_pix, self.roi.y_bottom, self.roi.y_top, self.yaxis.vmin, self.yaxis.vmax)
        else:
            y_val = self._map_log10(y_pix, self.roi.y_bottom, self.roi.y_top, self.yaxis.vmin, self.yaxis.vmax)

        x_val = self._snap(x_val, self.xaxis.ticks, self.xaxis.tick_tolerance)
        y_val = self._snap(y_val, self.yaxis.ticks, self.yaxis.tick_tolerance)
        return x_val, y_val

    def to_dataframe(self, detections, min_score: float = 0.4) -> pd.DataFrame:
        rows = []
        for score, flag, x_pix, y_pix in detections:
            if score < min_score:
                continue
            x_val, y_val = self.pixel_to_values(x_pix, y_pix)
            rows.append({
                "Score": score,
                "Flag": flag,
                "X_pix": x_pix, "Y_pix": y_pix,
                "X_val": x_val, "Y_val": y_val
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return df  # вернём пустой DataFrame без сортировки
        return df.sort_values(by="X_val")


def parse_chartreader_json(json_obj, image_key: str, category: str = "1"):
    """
    Возвращает список детекций [(score, flag, x_pix, y_pix), ...]
    Для нового формата, где категории — это индексы списка.
    """
    try:
        data = json_obj["result"][image_key]
    except KeyError:
        return []

    cat_idx = int(category)
    detections = []

    for group in data:  # group = [class0_list, class1_list, class2_list]
        if cat_idx < len(group):
            for p in group[cat_idx]:
                if len(p) >= 4:
                    detections.append((p[0], p[1], p[2], p[3]))

    return detections



