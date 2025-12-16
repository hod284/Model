from dataclasses import dataclass
from typing import List

import numpy as np
from ultralytics import YOLO

from .base_model import BaseModel


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


class DetectorModel(BaseModel):
    """
    YOLOv8 기반 번호판 Detector
    - predict(image): 이미지에서 번호판 bbox 리스트 반환
    - train(data_yaml, epochs, imgsz): YOLO 학습 래핑
    """

    def __init__(self, weights: str = "",expand_box: bool = True):
        """
        weights: 학습된 가중치 경로
        """
        self.weights_path = weights
        self.model = YOLO(weights)  # YOLOv8 모델 로드
        self.expand_box = expand_box

    def predict(self, image: np.ndarray) -> List[BBox]:
        """
        image: numpy (H, W, 3), BGR(OpenCV)
        return: BBox 리스트
        """
        results = self.model(image)

        boxes: List[BBox] = []
        r = results[0]

        h, w, _ = image.shape

        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            conf = float(b.conf[0])

            # 🔥 한국 번호판용 박스 확장
            if self.expand_box:
                width = x2 - x1
                height = y2 - y1

                # 좌우 12% 확장 (한글까지 포함)
                expand_x = width * 0.12
                x1 = max(0, x1 - expand_x)
                x2 = min(w, x2 + expand_x)

                # 상하 8% 확장
                expand_y = height * 0.08
                y1 = max(0, y1 - expand_y)
                y2 = min(h, y2 + expand_y)

            boxes.append(
                BBox(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    score=conf,
                )
            )

        # 신뢰도 필터 (필요에 따라 조절)
        boxes = [b for b in boxes if b.score > 0.4]

        return boxes

    def train(self, data_yaml: str, epochs: int = 100, imgsz: int = 640):
        """
        YOLOv8 학습
        data_yaml: YOLO data.yaml 경로
        """
        model = YOLO(self.weights_path)  # 예: 'yolov8n.pt' 같은 base

        model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
        )
        # 결과 가중치는 runs/detect/train*/weights/best.pt 등에 저장

    def load(self, path: str):
        self.model = YOLO(path)
        self.weights_path = path