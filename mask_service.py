import cv2
import numpy as np
import uuid
import os
import io
from s3_service import S3Service

class MaskService:
    def __init__(self, s3: S3Service):
        self.s3 = s3

    async def clear_graph(self, image_path: str, mask_path: str) -> str:
        # 1. Скачиваем исходное изображение
        img_bytes = self.s3.download_file(image_path)
        tmp_img_path = f"/tmp/tmp_img_{uuid.uuid4().hex}.png"
        with open(tmp_img_path, "wb") as f:
            f.write(img_bytes)

        # 2. Скачиваем маску
        mask_bytes = self.s3.download_file(mask_path)
        tmp_mask_path = f"/tmp/tmp_mask_{uuid.uuid4().hex}.png"
        with open(tmp_mask_path, "wb") as f:
            f.write(mask_bytes)

        # 3. Читаем через OpenCV
        img = cv2.imread(tmp_img_path)
        mask = cv2.imread(tmp_mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {tmp_img_path}")
        if mask is None:
            raise ValueError(f"Не удалось загрузить маску: {tmp_mask_path}")

        # 4. Бинаризация маски
        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        #Инвертирование: теперь закрашенные области будут "оставлены", а фон удалён
        mask_bin = cv2.bitwise_not(mask_bin)
        # 5. Проверка размеров
        if mask_bin.shape[:2] != img.shape[:2]:
            mask_bin = cv2.resize(mask_bin, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 6. Очистка
        result = cv2.inpaint(img, mask_bin, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        # 7. Сохраняем результат в память
        out_img = io.BytesIO()
        ok, buf = cv2.imencode(".png", result)
        if not ok:
            raise ValueError("Не удалось закодировать результат в PNG")
        out_img.write(buf.tobytes())
        out_img.seek(0)

        # 8. Загружаем в S3 — формируем ключ в папке mask
        s3_key = f"mask/graph_clean_{uuid.uuid4().hex}.png"
        s3_url = self.s3.upload_file(s3_key, out_img.read(), content_type="image/png")

        # 9. Убираем временные файлы
        os.remove(tmp_img_path)
        os.remove(tmp_mask_path)

        # 10. Возвращаем именно ключ/URL из S3
        return s3_url
