import os
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
import easyocr
from ultralytics import YOLO

from s3_service import S3Service


@dataclass
class AxisPoint:
    x: int
    y: int
    x_f: float
    y_f: float
    conf: float
    status: str


def _clamp_xy(x: float, y: float, w: int, h: int) -> Tuple[int, int]:
    xi = int(max(0, min(w - 1, round(float(x)))))
    yi = int(max(0, min(h - 1, round(float(y)))))
    return xi, yi


def _decode_image(img_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image bytes")
    return img


class YoloAxisService:
    def __init__(self, s3: S3Service, model_path: Optional[str] = None):
        self.s3 = s3
        self.default_model_path = model_path or os.getenv(
            "YOLO_AXIS_MODEL_PATH",
            os.path.join(os.path.dirname(__file__), "YOLO", "models", "axis", "best.pt"),
        )
        self._models: Dict[str, YOLO] = {}

    def _get_model(self, model_path: Optional[str]) -> YOLO:
        path = model_path or self.default_model_path
        if path not in self._models:
            self._models[path] = YOLO(path)
        return self._models[path]

    def predict_axis(
        self,
        image_path: str,
        conf: float = 0.25,
        kpt_conf: float = 0.25,
        model_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        img_bytes = self.s3.download_file(image_path)
        img = _decode_image(img_bytes)
        h, w = img.shape[:2]

        model = self._get_model(model_path)
        results = model.predict(source=img, conf=conf, verbose=False)
        r = results[0] if results else None

        if r is None or r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            raise ValueError("No detections found")

        scores = r.boxes.conf.cpu().numpy()
        best_i = int(np.argmax(scores))

        kpts_xy = r.keypoints.xy.cpu().numpy()[best_i]
        kpts_cf = r.keypoints.conf.cpu().numpy()[best_i]

        names = ["origin", "x_end", "y_end"]
        axis: Dict[str, AxisPoint] = {}

        for j, name in enumerate(names):
            x_f, y_f = float(kpts_xy[j][0]), float(kpts_xy[j][1])
            c = float(kpts_cf[j])
            x_i, y_i = _clamp_xy(x_f, y_f, w, h)
            status = "ok" if c >= kpt_conf else "low_conf"
            axis[name] = AxisPoint(x=x_i, y=y_i, x_f=x_f, y_f=y_f, conf=c, status=status)

        return {
            "image": {"width": w, "height": h},
            "axis": {k: axis[k].__dict__ for k in names},
            "best_bbox_conf": float(scores[best_i]),
            "model": {
                "path": model_path or self.default_model_path,
                "conf": conf,
                "kpt_conf": kpt_conf,
            },
        }


_num_re = re.compile(r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?")
_simple_float_re = re.compile(r"^[-+]?\d+(?:\.\d+)?$")
_sci_re = re.compile(r"^[-+]?\d+(?:\.\d+)?[eE][-+]?\d+$")
_pow10_re = re.compile(r"^([-+]?\d+(?:\.\d+)?)[*]?10(?:\^|\\*\\*)([-+]?\d+)$")
_pow10_only_re = re.compile(r"^10(?:\^|\\*\\*)([-+]?\d+)$")
_frac_re = re.compile(r"^([-+]?\d+(?:\.\d+)?)/([-+]?\d+(?:\.\d+)?)$")
_mul_re = re.compile(r"^([-+]?\d+(?:\.\d+)?)[*]([-+]?\d+(?:\.\d+)?)$")

_sup_map = {
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁻": "-",
    "⁺": "+",
}
_frac_map = {
    "½": "0.5",
    "¼": "0.25",
    "¾": "0.75",
    "⅓": "0.3333333333",
    "⅔": "0.6666666667",
}


def _normalize_numeric_text(s: str) -> str:
    if not s:
        return ""
    t = s.strip().replace(" ", "")
    t = t.replace(",", ".")
    t = t.replace("×", "*").replace("·", "*")
    for k, v in _frac_map.items():
        t = t.replace(k, v)

    # common OCR: "x10" / "X10" as multiplication
    t = re.sub(r"(\d)[xX]10", r"\\1*10", t)
    t = re.sub(r"(\d)[xX]", r"\\1*", t)

    # convert superscripts to ^exp (e.g., 10⁻³ -> 10^-3)
    out: List[str] = []
    i = 0
    while i < len(t):
        ch = t[i]
        if ch in _sup_map:
            exp = []
            while i < len(t) and t[i] in _sup_map:
                exp.append(_sup_map[t[i]])
                i += 1
            out.append("^" + "".join(exp))
            continue
        out.append(ch)
        i += 1
    t = "".join(out)
    return t


def _parse_number(s: str) -> Optional[float]:
    if not s:
        return None
    t = _normalize_numeric_text(s)
    if not t:
        return None

    if _simple_float_re.match(t):
        try:
            return float(t)
        except Exception:
            return None

    if _sci_re.match(t):
        try:
            return float(t)
        except Exception:
            return None

    m = _pow10_re.match(t)
    if m:
        try:
            base = float(m.group(1))
            exp = int(m.group(2))
            return base * (10.0 ** exp)
        except Exception:
            return None

    m = _pow10_only_re.match(t)
    if m:
        try:
            exp = int(m.group(1))
            return 10.0 ** exp
        except Exception:
            return None

    m = _frac_re.match(t)
    if m:
        try:
            num = float(m.group(1))
            den = float(m.group(2))
            if den == 0:
                return None
            return num / den
        except Exception:
            return None

    m = _mul_re.match(t)
    if m:
        try:
            return float(m.group(1)) * float(m.group(2))
        except Exception:
            return None

    # fallback: extract first float-like token
    m = _num_re.search(t)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", "."))
    except Exception:
        return None


_easyocr_readers: Dict[Tuple[Tuple[str, ...], bool], easyocr.Reader] = {}


def _get_easyocr_reader(langs: List[str], gpu: bool) -> easyocr.Reader:
    key = (tuple(langs), bool(gpu))
    if key not in _easyocr_readers:
        _easyocr_readers[key] = easyocr.Reader(langs, gpu=gpu)
    return _easyocr_readers[key]


def _ocr_crop(
    rgb_img: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    pad: int,
    scale: int,
    reader: easyocr.Reader,
    allowlist: str,
) -> str:
    h, w = rgb_img.shape[:2]
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(w - 1, int(x2) + pad)
    y2 = min(h - 1, int(y2) + pad)

    crop = rgb_img[y1:y2, x1:x2]
    if crop.size == 0:
        return ""

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    results = reader.readtext(thr, allowlist=allowlist, detail=0, paragraph=False)
    if not results:
        return ""

    txt = str(results[0]).strip()
    txt = txt.replace(" ", "").replace("\n", "")
    return txt


def _cluster_ticks_by_pos(ticks: List[Tuple[float, float, str, Tuple[float, float, float, float], float]], pos_eps: float) -> List[Tuple[float, float, str, Tuple[float, float, float, float], float]]:
    if not ticks:
        return []

    ticks = sorted(ticks, key=lambda t: t[0])
    clusters: List[List[Tuple[float, float, str, Tuple[float, float, float, float], float]]] = []
    cur = [ticks[0]]

    for t in ticks[1:]:
        if abs(t[0] - cur[-1][0]) <= pos_eps:
            cur.append(t)
        else:
            clusters.append(cur)
            cur = [t]
    clusters.append(cur)

    clean = []
    for c in clusters:
        best = max(c, key=lambda t: t[4])
        clean.append(best)

    return clean


def _fix_minus_by_context(
    x_ticks_sorted: List[Tuple[float, float, str, Tuple[float, float, float, float], float]]
) -> List[Tuple[float, float, str, Tuple[float, float, float, float], float]]:
    if len(x_ticks_sorted) < 3:
        return x_ticks_sorted

    out = [x_ticks_sorted[0]]
    for t in x_ticks_sorted[1:]:
        prev = out[-1]
        prev_v = prev[1]
        v = t[1]

        if v >= prev_v:
            out.append(t)
            continue

        raw = t[2] or ""
        if "-" not in raw and v != 0:
            v2 = -v
            if v2 >= prev_v:
                out.append((t[0], float(v2), raw, t[3], t[4]))
                continue

        continue

    return out


class YoloTicksService:
    def __init__(self, s3: S3Service, model_path: Optional[str] = None):
        self.s3 = s3
        self.default_model_path = model_path or os.getenv(
            "YOLO_TICKS_MODEL_PATH",
            os.path.join(os.path.dirname(__file__), "YOLO", "models", "ticks", "best.pt"),
        )
        self._models: Dict[str, YOLO] = {}
        langs_raw = os.getenv("EASYOCR_LANGS", "en")
        self.ocr_langs = [s.strip() for s in langs_raw.split(",") if s.strip()]
        self.ocr_gpu = os.getenv("EASYOCR_GPU", "0") == "1"
        self.ocr_allowlist = os.getenv("EASYOCR_ALLOWLIST", "0123456789.,-+eE")

    def _get_model(self, model_path: Optional[str]) -> YOLO:
        path = model_path or self.default_model_path
        if path not in self._models:
            self._models[path] = YOLO(path)
        return self._models[path]

    def predict_ticks(
        self,
        image_path: str,
        conf: float = 0.25,
        iou: float = 0.6,
        conf_min: float = 0.1,
        class_x: int = 0,
        class_y: int = 1,
        pos_eps: float = 12.0,
        ocr_pad: int = 2,
        ocr_scale: int = 4,
        fix_minus: bool = True,
        model_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        img_bytes = self.s3.download_file(image_path)
        img_bgr = _decode_image(img_bytes)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        model = self._get_model(model_path)
        res = model.predict(img_bgr, conf=conf, iou=iou, verbose=False)[0]

        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            raise ValueError("No detections")

        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        x_ticks: List[Tuple[float, float, str, Tuple[float, float, float, float], float]] = []
        y_ticks: List[Tuple[float, float, str, Tuple[float, float, float, float], float]] = []

        reader = _get_easyocr_reader(self.ocr_langs, self.ocr_gpu)

        for i in range(len(xyxy)):
            if float(confs[i]) < conf_min:
                continue
            c = int(cls[i])
            x1, y1, x2, y2 = xyxy[i]
            raw = _ocr_crop(
                img_rgb,
                x1,
                y1,
                x2,
                y2,
                pad=ocr_pad,
                scale=ocr_scale,
                reader=reader,
                allowlist=self.ocr_allowlist,
            )
            val = _parse_number(raw)
            if val is None:
                continue

            x_center = float((x1 + x2) / 2.0)
            y_center = float((y1 + y2) / 2.0)

            if c == class_x:
                x_ticks.append((x_center, float(val), raw, (float(x1), float(y1), float(x2), float(y2)), float(confs[i])))
            elif c == class_y:
                y_ticks.append((y_center, float(val), raw, (float(x1), float(y1), float(x2), float(y2)), float(confs[i])))

        if not x_ticks or not y_ticks:
            raise ValueError("No readable ticks after filtering/OCR")

        x_ticks_clean = _cluster_ticks_by_pos(x_ticks, pos_eps=pos_eps)
        y_ticks_clean = _cluster_ticks_by_pos(y_ticks, pos_eps=pos_eps)

        x_ticks_clean = sorted(x_ticks_clean, key=lambda t: t[0])
        y_ticks_clean = sorted(y_ticks_clean, key=lambda t: t[0])

        if fix_minus:
            x_ticks_clean = _fix_minus_by_context(x_ticks_clean)

        if len(x_ticks_clean) < 2 or len(y_ticks_clean) < 2:
            raise ValueError("Not enough clean ticks to determine axis bounds")

        x_min = float(x_ticks_clean[0][1])
        x_max = float(x_ticks_clean[-1][1])
        y_max = float(y_ticks_clean[0][1])
        y_min = float(y_ticks_clean[-1][1])

        origin = {"x": x_min, "y": y_min}

        def _pack_tick(t):
            return {
                "pos": float(t[0]),
                "value": float(t[1]),
                "raw": t[2],
                "bbox": [float(t[3][0]), float(t[3][1]), float(t[3][2]), float(t[3][3])],
                "conf": float(t[4]),
            }

        return {
            "origin": origin,
            "x_end": x_max,
            "y_end": y_max,
            "x_ticks": [_pack_tick(t) for t in x_ticks_clean],
            "y_ticks": [_pack_tick(t) for t in y_ticks_clean],
            "stats": {
                "x_ticks_raw": len(x_ticks),
                "y_ticks_raw": len(y_ticks),
                "x_ticks_clean": len(x_ticks_clean),
                "y_ticks_clean": len(y_ticks_clean),
            },
            "model": {
                "path": model_path or self.default_model_path,
                "conf": conf,
                "iou": iou,
                "conf_min": conf_min,
                "class_x": class_x,
                "class_y": class_y,
            },
        }
