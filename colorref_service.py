import json
from typing import Dict, Any, Optional, List
from fastapi import UploadFile
from s3_service import S3Service
from chart_processor import ChartProcessorService, parse_chartreader_json
from calibration_service import ROI, AxisConfig, PlotCalibrator

class ColorRefService:
    def __init__(self, s3: S3Service):
        self.s3 = s3

    async def process(self,
                      markup: Dict[str, Any],
                      calibration: Dict[str, Any],
                      image_path: str,
                      refs: List[Dict[str, str]],
                      file: Optional[UploadFile] = None) -> Dict[str, Any]:

        # сохранить файл если пришёл
        local_path = "/tmp/colorref.png"
        if file:
            img_bytes = await file.read()
            key = f"uploads/{file.filename}"
            self.s3.upload_file(key, img_bytes, content_type="image/png")
            with open(local_path, "wb") as f:
                f.write(img_bytes)
            image_path = key
        else:
            img_bytes = self.s3.download_file(image_path)
            with open(local_path, "wb") as f:
                f.write(img_bytes)

        # калибратор
        calibrator = None
        try:
            roi = ROI(
                x_left=min(calibration["x1_pix"], calibration["x2_pix"]),
                x_right=max(calibration["x1_pix"], calibration["x2_pix"]),
                y_top=min(calibration["y1_pix_axis"], calibration["y2_pix_axis"]),
                y_bottom=max(calibration["y1_pix_axis"], calibration["y2_pix_axis"]),
            )
            xaxis = AxisConfig(vmin=calibration["x1_val"], vmax=calibration["x2_val"])
            yaxis = AxisConfig(vmin=calibration["y1_val"], vmax=calibration["y2_val"])
            calibrator = PlotCalibrator(roi, xaxis, yaxis)
        except Exception:
            pass

        # reference colors
        reference_colors = {ref["name"]: tuple(map(int, ref["rgb"].split(","))) for ref in refs}

        # точки из markup
        image_key = list(markup["result"].keys())[0]
        detections = parse_chartreader_json(markup, image_key, category="1")
        points = [(int(x), int(y)) for score, flag, x, y in detections if score >= 0.4]

        processor = ChartProcessorService(self.s3, calibrator)
        return processor.process_colorrefs(
            image_path=image_path,
            local_path=local_path,
            points=points,
            reference_colors=reference_colors,
            threshold=20.0
        )
