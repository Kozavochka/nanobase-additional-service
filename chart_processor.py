import cv2, io
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from calibration_service import PlotCalibrator, parse_chartreader_json
from s3_service import S3Service
from sklearn.cluster import DBSCAN
from skimage.color import rgb2lab
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt

def cluster_points_by_color(
    image_path: str,
    points: list[tuple[int, int]],
    patch_radius: int = 1,
    dbscan_eps: float = 5.0,
    dbscan_min_samples: int = 1
) -> dict[int, list[tuple[int, int]]]:
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")
    h, w, _ = img_bgr.shape

    samples = []
    for x, y in points:
        x0, y0 = int(x), int(y)
        x1, x2 = max(0, x0 - patch_radius), min(w, x0 + patch_radius + 1)
        y1, y2 = max(0, y0 - patch_radius), min(h, y0 + patch_radius + 1)
        patch = img_bgr[y1:y2, x1:x2]
        samples.append(patch.reshape(-1, 3).mean(axis=0))
    samples = np.array(samples, dtype=np.float32)

    samples_rgb = samples[:, ::-1] / 255.0
    samples_lab = rgb2lab(samples_rgb)

    clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(samples_lab)
    labels = clustering.labels_

    clusters: dict[int, list[tuple[int, int]]] = {}
    for lbl, pt in zip(labels, points):
        clusters.setdefault(int(lbl), []).append(pt)

    return clusters

class ChartProcessorService:
    def __init__(self, s3: S3Service, calibrator: PlotCalibrator):
        self.s3 = s3
        self.calibrator = calibrator

    def process(self, image_path: str, markup: dict, calibration: dict, min_score: float = 0.4):
        # 1. Скачиваем исходное изображение
        img_bytes = self.s3.download_file(image_path)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # 2. Парсим детекции
        image_key = list(markup["result"].keys())[0]
        detections = parse_chartreader_json(markup, image_key, category="1")

        # 3. Наносим точки на изображение
        draw = ImageDraw.Draw(img)
        for score, flag, x, y in detections:
            if score < min_score:
                continue
            color = "red" if flag == 1 else "blue"
            r = 4
            draw.ellipse((x-r, y-r, x+r, y+r), outline=color, width=2)

        # 4. Сохраняем размеченное изображение в S3
        out_img = io.BytesIO()
        img.save(out_img, format="PNG")
        out_img.seek(0)
        annotated_key = image_path.replace("uploads/", "processed/annotated_")
        annotated_url = self.s3.upload_file(annotated_key, out_img.read(), content_type="image/png")

        # 5. Преобразуем в значения и сохраняем Excel
        df = self.calibrator.to_dataframe(detections, min_score=min_score)
        out_excel = io.BytesIO()
        with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        out_excel.seek(0)
        excel_key = image_path.replace("uploads/", "processed/values_").replace(".png", ".xlsx")
        excel_url = self.s3.upload_file(excel_key, out_excel.read(), content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        return {
            "annotated_image": annotated_url,
            "excel_file": excel_url
        }
    

    def process_dbscan(
            self,
            image_path: str,
            markup: dict,
            calibration: dict,
            eps_setting: float = 15.0,
            min_score: float = 0.4
        ) -> Dict[str, Any]:
            # 1. Скачиваем исходное изображение
            img_bytes = self.s3.download_file(image_path)
            with open("/tmp/tmp_img.png", "wb") as f:
                f.write(img_bytes)
            local_path = "/tmp/tmp_img.png"

            # 2. Парсим детекции
            image_key = list(markup["result"].keys())[0]
            detections = parse_chartreader_json(markup, image_key, category="1")

            points = [(int(x), int(y)) for score, flag, x, y in detections if score >= min_score]

            # 3. Кластеризация
            clusters = cluster_points_by_color(local_path, points, dbscan_eps=eps_setting)

            # 4. Визуализация через matplotlib
            img_bgr = cv2.imread(local_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(10, 6))
            plt.imshow(img_rgb, origin="upper")
            plt.axis("off")

            cmap = plt.get_cmap("tab10")
            for lbl, pts in clusters.items():
                pts_arr = np.array(pts)
                if len(pts_arr) < 3:
                    continue
                # цвет тоже по номеру кластера
                color = (0, 0, 0) if lbl == -1 else cmap(lbl % 10)[:3]
                plt.scatter(
                    pts_arr[:, 0],
                    pts_arr[:, 1],
                    c=[color],
                    s=50,
                    edgecolors="white",
                    linewidths=0.5,
                    label=f"Cluster {lbl}"
                )

            plt.legend(loc="upper right", fontsize="small", framealpha=0.7)
            plt.tight_layout()

            out_img = io.BytesIO()
            plt.savefig(out_img, format="png", dpi=150)
            out_img.seek(0)
            annotated_key = image_path.replace("uploads/", "processed/dbscan_")
            annotated_url = self.s3.upload_file(annotated_key, out_img.read(), content_type="image/png")
            plt.close()

            # 5. Excel с листами по кластерам
            out_excel = io.BytesIO()
            with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
                for lbl, pts in clusters.items():
                    # переводим в значения осей
                    dets = [(1.0, 0, x, y) for (x, y) in pts]  # score/flag фиктивные
                    df = self.calibrator.to_dataframe(dets, min_score=0.0)
                    df.to_excel(writer, sheet_name=f"Cluster_{lbl}", index=False)
            out_excel.seek(0)
            excel_key = image_path.replace("uploads/", "processed/dbscan_values_").replace(".png", ".xlsx")
            excel_url = self.s3.upload_file(excel_key, out_excel.read(),
                                            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            return {
                "annotated_clusters_image": annotated_url,
                "excel_file": excel_url,
                "clusters_count": len(clusters)
            }
