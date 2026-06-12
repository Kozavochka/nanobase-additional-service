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

    @staticmethod
    def chartreader_groups_to_clusters(
        chartreader_groups: List[List[float]],
        category_idx: int = 0,
    ) -> Dict[str, List[List[int]]]:
        clusters: Dict[str, List[List[int]]] = {}
        cluster_id = 0

        for group in chartreader_groups:
            if not isinstance(group, list) or len(group) < 6:
                continue

            try:
                group_category = int(group[-1])
            except (TypeError, ValueError):
                continue

            if group_category != int(category_idx):
                continue

            # group format: [cx, cy, x1, y1, x2, y2, ..., score, class]
            keypoint_coords = group[2:-2]
            if len(keypoint_coords) < 4:
                continue

            points: List[List[int]] = []
            for idx in range(0, len(keypoint_coords) - 1, 2):
                try:
                    x = int(round(float(keypoint_coords[idx])))
                    y = int(round(float(keypoint_coords[idx + 1])))
                except (TypeError, ValueError):
                    continue
                points.append([x, y])

            if len(points) < 2:
                continue

            clusters[str(cluster_id)] = points
            cluster_id += 1

        return clusters

    @staticmethod
    def merge_clusters_by_endpoint_distance(
        clusters: Dict[str, List[List[int]]],
        post_merge_distance_px: float = 18.0,
    ) -> Dict[str, List[List[int]]]:
        prepared: List[List[Tuple[int, int]]] = []
        for pts in clusters.values():
            if not pts:
                continue
            unique_pts = {(int(p[0]), int(p[1])) for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2}
            if len(unique_pts) < 2:
                continue
            prepared.append(sorted(unique_pts, key=lambda p: (p[0], p[1])))

        if len(prepared) <= 1:
            return {str(i): [[x, y] for x, y in pts] for i, pts in enumerate(prepared)}

        threshold = float(post_merge_distance_px)
        merged_any = True
        max_iters = 300
        iters = 0

        def _dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            dx = float(a[0] - b[0])
            dy = float(a[1] - b[1])
            return float((dx * dx + dy * dy) ** 0.5)

        while merged_any and iters < max_iters:
            iters += 1
            merged_any = False

            for i in range(len(prepared)):
                if merged_any:
                    break
                for j in range(i + 1, len(prepared)):
                    a = prepared[i]
                    b = prepared[j]
                    a_left, a_right = a[0], a[-1]
                    b_left, b_right = b[0], b[-1]

                    # Merge only "right-to-left" endpoints to avoid joining parallel lines.
                    d_ar_bl = _dist(a_right, b_left)
                    d_br_al = _dist(b_right, a_left)

                    can_merge_ar_bl = (b_left[0] >= a_right[0]) and (d_ar_bl <= threshold)
                    can_merge_br_al = (a_left[0] >= b_right[0]) and (d_br_al <= threshold)

                    if not can_merge_ar_bl and not can_merge_br_al:
                        continue

                    merged_pts = set(a) | set(b)
                    prepared[i] = sorted(merged_pts, key=lambda p: (p[0], p[1]))
                    prepared.pop(j)
                    merged_any = True
                    break

        return {str(i): [[x, y] for x, y in pts] for i, pts in enumerate(prepared)}

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
