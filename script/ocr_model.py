from typing import List, Tuple, Any
import numpy as np
import cv2
import easyocr
import re
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
        # 그레이스케일
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        
        # 크기키우기
        h, w = gray.shape
        if h < 100:
            scale = 100 / h
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
        #노이즈 제거  
        denoised = cv2.fastNlMeansDenoising(gray,h=10)
        # 4. 샤프닝 (선명도 향상!)
        kernel = np.array([ [-1, -1, -1], [-1,  9, -1],[-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
    
        # 5. CLAHE (명암 대비)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(sharpened)
    
        # 6. 이진화 (흑백 명확히)
        _, binary = cv2.threshold(enhanced, 0, 255, 
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
        # 7. 모폴로지 (글자 두껍게)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel2)
    
        # 8. 패딩
        padded = cv2.copyMakeBorder(morphed, 10, 10, 10, 10,
                                cv2.BORDER_CONSTANT, value=255)
        return padded

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

        # detail=1 → 텍스트만 리스트로 반환
        results = self.reader.readtext(
           processed, 
           detail=0,  # ← 변경
        )

        if not results:
            return ""
        best = max(results, key=len)
        best = best.replace(" ", "")
        # ✅ 오인식 보정 (추가!)
        best = self.correct_ocr_errors(best)
        return best

    def train(self, *args, **kwargs):
        raise NotImplementedError("easyocr 자체 학습은 여기서 지원하지 않습니다.")
    
    def correct_ocr_errors(self, text):
        """
        OCR 오인식 보정
        """
        # 1. 영어 → 숫자 보정
        corrections = {
           'O': '0',  # O → 0
           'o': '0',
           'I': '1',  # I → 1
           'l': '1',
           'Z': '2',
           'S': '5',
           'B': '8',
           'g': '9',
       }
    
        result = ""
        for i, char in enumerate(text):
          # 숫자 위치에서만 보정
           if i < 2 or i > len(text) - 5:  # 앞 2자리, 뒤 4자리
                if char in corrections:
                   result += corrections[char]
                else:
                    result += char
           else:
             result += char
    
         # 2. 한글 자모 결합
        result = self.combine_korean(result)
    
         # 3. 한국 번호판 형식 검증
        patterns = [
            # 1. 일반 자동차 (가장 흔함)
            r'\d{2,3}[가-힣]\d{4}',
            
            # 2. 오토바이/특수 (지역명 + 숫자 + 한글 + 숫자)
            r'[가-힣]{2,3}\d{1,2}[가-힣]\d{4}',
            
            # 3. 영업용 (지역명 + 숫자 + 한글 + 숫자)
            r'[가-힣]{2}\d{2,3}[가-힣]\d{4}',
        ]
        
    
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0]
        return ""

    def combine_korean(self, text):
        """
        한글 자모 결합
        예: "ㄱㅏ" → "가"
        """
        # 간단 버전 (라이브러리 사용 권장)
        # pip install korean-romanizer
    
        # 또는 수동 구현
        result = text
    
         # ㄱㅏ → 가
        result = result.replace('ㄱㅏ', '가')
        result = result.replace('ㄴㅏ', '나')
        result = result.replace('ㄷㅏ', '다')
         # ... (필요한 조합 추가)
    
        return result