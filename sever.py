import cv2
import numpy as np
import grpc
from concurrent import futures

import plate_recognizer_pb2 as pb2
import plate_recognizer_pb2_grpc as pb2_grpc

from script.detector_model import DetectorModel
from script.ocr_model import OcrModel

class PlateRecognizerService(pb2_grpc.PlateRecognizerServicer):
    def __init__(self):
        # YOLO 번호판 탐지 모델
        self.detector = DetectorModel()

        # easyocr OCR
        self.ocr = OcrModel(use_dummy=False, languages=['ko', 'en'])

    def Recognize(self, request, context):
        # 0) 모드 읽기 (enum, 안 보내면 기본값 0 = MODE_FULL)
        mode = request.mode

        # 1) bytes → numpy → OpenCV 이미지
        image_bytes = request.image
        if not image_bytes:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Empty image bytes")
            return pb2.PlateResponse()

        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Cannot decode image")
            return pb2.PlateResponse()

        # 2) 모드에 따라서 서로 다른 로직 실행
        if mode == pb2.MODE_FULL:
            # 탐지 + OCR 모두
            return self._run_full_pipeline(img)

        elif mode == pb2.MODE_DETECT_ONLY:
            # 박스만 찾기 (OCR 안 함)
            return self._run_detect_only(img)

        elif mode == pb2.MODE_OCR_ONLY:
            # OCR만 (이미 번호판만 잘린 이미지라고 가정)
            return self._run_ocr_only(img)

        else:
            # 정의되지 않은 모드
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"Unknown mode: {mode}")
            return pb2.PlateResponse()

    # === 모드별 실제 처리 함수 ===

    def _run_full_pipeline(self, img):
        """
        1) YOLO로 번호판 박스 탐지
        2) 가장 점수 높은 박스 선택
        3) crop해서 OCR 수행
        """
        boxes = self.detector.predict(img)
        if not boxes:
            return pb2.PlateResponse(
                plate_number="",
                x=0.0, y=0.0, width=0.0, height=0.0,
            )

        # score 가장 높은 박스 선택
        box = sorted(boxes, key=lambda b: b.score, reverse=True)[0]

        h, w, _ = img.shape
        x1 = int(max(box.x1, 0))
        y1 = int(max(box.y1, 0))
        x2 = int(min(box.x2, w))
        y2 = int(min(box.y2, h))

        # 번호판 영역 crop
        plate_img = img[y1:y2, x1:x2]

        # OCR
        plate_number = self.ocr.predict(plate_img)

        return pb2.PlateResponse(
            plate_number=plate_number,
            x=float(x1),
            y=float(y1),
            width=float(x2 - x1),
            height=float(y2 - y1),
        )

    def _run_detect_only(self, img):
        """
        YOLO로 번호판 박스만 찾고,
        OCR은 하지 않고 좌표만 리턴
        """
        boxes = self.detector.predict(img)
        if not boxes:
            return pb2.PlateResponse(
                plate_number="",
                x=0.0, y=0.0, width=0.0, height=0.0,
            )

        box = sorted(boxes, key=lambda b: b.score, reverse=True)[0]

        h, w, _ = img.shape
        x1 = int(max(box.x1, 0))
        y1 = int(max(box.y1, 0))
        x2 = int(min(box.x2, w))
        y2 = int(min(box.y2, h))

        return pb2.PlateResponse(
            plate_number="",  # OCR 안했으니까 빈 문자열
            x=float(x1),
            y=float(y1),
            width=float(x2 - x1),
            height=float(y2 - y1),
        )

    def _run_ocr_only(self, img):
        """
        OCR만 수행하는 경우.
        여기서는 '이미 번호판만 잘린 이미지' 라고 가정하고
        전체 이미지에 대해 OCR만 적용.
        좌표는 의미 없으니까 0으로 채움.
        """
        plate_number = self.ocr.predict(img)

        return pb2.PlateResponse(
            plate_number=plate_number,
            x=0.0,
            y=0.0,
            width=0.0,
            height=0.0,
        )


def serve():
    # gRPC 서버 생성 + 서비스 등록
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb2_grpc.add_PlateRecognizerServicer_to_server(
        PlateRecognizerService(),
        server
    )


    # 50051 포트에서 대기
    server.add_insecure_port("[::]:50051")
    print("PlateRecognizer gRPC server started on port 50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()