import numpy as np
import cv2
import easyocr

from .base_model import BaseModel


class OcrModel(BaseModel):
    """
    번호판 OCR 모델 (easyocr 사용)
    """

    def __init__(self, use_dummy: bool = False, languages=None):
        """
        use_dummy: True면 테스트용 더미 문자열 반환
        languages: easyocr 사용 언어 리스트 (기본: 한국어 + 영어)
        """
        self.use_dummy = use_dummy

        if languages is None:
            languages = ['ko', 'en']

        if not use_dummy:
            self.reader = easyocr.Reader(languages, gpu=False)
        else:
            self.reader = None
    # -> 이표시는 주석으로 이렇게 하세요라는 표시
    def preprocess(self, plate_image: np.ndarray) -> np.ndarray:
        """
        OCR 전처리
        - BGR → GRAY
        - 필요시 이진화 등 추가
        """
        if plate_image is None or plate_image.size == 0:
            return plate_image

        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        # 필요하면 이진화:
        # _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return gray

    def predict(self, plate_image: np.ndarray) -> str:
        """
        plate_image: 번호판이 잘려 있는 이미지 (BGR)
        return: 인식된 번호판 문자열 (없으면 "")
        """
        if self.use_dummy:
            return "12가3456"

        if self.reader is None:
            raise RuntimeError("easyocr Reader가 초기화되지 않았습니다.")

        if plate_image is None or plate_image.size == 0:
            return ""

        processed = self.preprocess(plate_image)

        # detail=0 → 텍스트만 리스트로 반환
        results = self.reader.readtext(processed, detail=0)

        if not results:
            return ""

        # 가장 길어 보이는 문자열 선택
        best = max(results, key=len)
        best = best.replace(" ", "")

        return best

    def train(self, *args, **kwargs):
        raise NotImplementedError("easyocr 자체 학습은 여기서 지원하지 않습니다.")