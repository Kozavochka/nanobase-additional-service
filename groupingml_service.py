import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calibration_service import PlotCalibrator
from s3_service import S3Service


class GroupingMLService:
    def __init__(self, s3: S3Service, calibrator: Optional[PlotCalibrator]):
        self.s3 = s3
        self.calibrator = calibrator

    def process(
        self,
        image_path: str,
        clusters: Dict[str, List[List[int]]],
        cluster_pixel_values: Optional[Dict[str, List[List[int]]]] = None,
        noise: Optional[List[List[int]]] = None,
        min_points_visual: int = 3,
    ) -> Dict[str, Any]:
        img_bytes = self.s3.download_file(image_path)
        local_path = "/tmp/groupingml.png"
        with open(local_path, "wb") as f:
            f.write(img_bytes)

        img_bgr = cv2.imread(local_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to load image from S3 key: {image_path}")
        draw_bgr = img_bgr.copy()
        point_radius = 3

        cmap = plt.get_cmap("tab20")
        for idx, (cluster_id, pts) in enumerate(clusters.items()):
            if not pts:
                continue

            pts_arr = np.array(pts)
            if len(pts_arr) < min_points_visual:
                continue

            color = None
            if cluster_pixel_values and cluster_id in cluster_pixel_values and cluster_pixel_values[cluster_id]:
                px = np.array(cluster_pixel_values[cluster_id], dtype=np.float32)
                mean_rgb = np.clip(px.mean(axis=0), 0, 255) / 255.0
                color = (float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2]))
            if color is None:
                color = cmap(idx % 20)[:3]

            color_bgr = (
                int(round(color[2] * 255)),
                int(round(color[1] * 255)),
                int(round(color[0] * 255)),
            )
            for x, y in pts_arr:
                cv2.circle(
                    draw_bgr,
                    (int(x), int(y)),
                    point_radius,
                    color_bgr,
                    -1,
                    lineType=cv2.LINE_AA,
                )

        ok, encoded = cv2.imencode(".png", draw_bgr)
        if not ok:
            raise RuntimeError("Failed to encode annotated image")
        out_img = io.BytesIO(encoded.tobytes())

        annotated_key = self._build_processed_key(image_path, "groupingml_")
        annotated_url = self.s3.upload_file(annotated_key, out_img.read(), content_type="image/png")

        out_excel = io.BytesIO()
        with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
            for cluster_id, pts in clusters.items():
                detections = [(1.0, 0, int(x), int(y)) for x, y in pts]

                if self.calibrator:
                    df = self.calibrator.to_dataframe(detections, min_score=0.0)
                else:
                    df = pd.DataFrame(
                        [
                            {"Score": score, "Flag": flag, "X_pix": x, "Y_pix": y}
                            for (score, flag, x, y) in detections
                        ]
                    )

                sheet_name = self._safe_sheet_name(f"Cluster_{cluster_id}")
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        out_excel.seek(0)
        excel_key = self._build_processed_excel_key(image_path)
        excel_url = self.s3.upload_file(
            excel_key,
            out_excel.read(),
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        return {
            "annotated_image": annotated_url,
            "excel_file": excel_url,
            "clusters_count": len(clusters),
        }

    def _build_processed_key(self, image_path: str, prefix: str) -> str:
        if image_path.startswith("uploads/"):
            return image_path.replace("uploads/", f"processed/{prefix}", 1)

        name = Path(image_path).name
        return f"processed/{prefix}{name}"

    def _build_processed_excel_key(self, image_path: str) -> str:
        if image_path.startswith("uploads/"):
            key = image_path.replace("uploads/", "processed/groupingml_values_", 1)
        else:
            key = f"processed/groupingml_values_{Path(image_path).name}"

        path = Path(key)
        return str(path.with_suffix(".xlsx"))

    def _safe_sheet_name(self, name: str) -> str:
        invalid = set('[]:*?/\\')
        cleaned = ''.join(ch for ch in name if ch not in invalid).strip()
        if not cleaned:
            cleaned = "Cluster"
        return cleaned[:31]
