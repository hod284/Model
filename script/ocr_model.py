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

    def preprocess(self, plate_image: np.ndarray) -> np.ndarray:
        """
        OCR 기본 전처리 - 가벼운 전처리만
        """
        if plate_image is None or plate_image.size == 0:
            return plate_image
        
        # 1. 그레이스케일
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        
        # 2. 크기 키우기
        h, w = gray.shape
        target_height = 150
        if h < target_height:
            scale = target_height / h
            gray = cv2.resize(gray, None, fx=scale, fy=scale, 
                            interpolation=cv2.INTER_CUBIC)
            print(f"[OCR] 크기 조정: {h}px → {target_height}px")
        
        # 3. CLAHE만 (명암 대비)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        return enhanced

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

        # ✅ 여러 전처리 방법 시도
        preprocess_methods = [
            ("기본", self.preprocess),
            ("샤프 강화", self.preprocess_sharp),
            ("이진화 강화", self.preprocess_binary),
        ]
        
        candidates = []  # 후보 번호판들
        
        for method_name, preprocess_func in preprocess_methods:
            try:
                processed = preprocess_func(plate_image)
                results = self.reader.readtext(processed, detail=0)
                
                if not results:
                    continue
                
                # 모든 텍스트 합치기 (타입 명시)
                all_text = "".join(str(r) for r in results).replace(" ", "")
                
                # 패턴 추출 시도
                plate = self.extract_korean_plate(all_text)
                
                if plate:
                    print(f"[OCR] {method_name} 방법으로 인식: {plate}")
                    candidates.append(plate)
            except Exception as e:
                print(f"[OCR] {method_name} 실패: {e}")
                continue
        
        # ✅ 후보가 있으면 가장 많이 나온 것 선택
        if candidates:
            from collections import Counter
            most_common = Counter(candidates).most_common(1)[0][0]
            print(f"[OCR] 최종 선택: {most_common} (총 {len(candidates)}개 후보)")
            return most_common
        
        # ✅ 패턴 못 찾으면 기본 방법으로 한 번 더
        print("[OCR] 패턴 매칭 실패, 기본 OCR 결과 반환")
        processed = self.preprocess(plate_image)
        results = self.reader.readtext(processed, detail=0)
        
        if not results:
            return ""
        
        best = max(results, key=len)
        best = best.replace(" ", "")
        best = self.correct_ocr_errors(best)
        
        return best
    
    def preprocess_sharp(self, plate_image: np.ndarray) -> np.ndarray:
        """
        샤프 강화 전처리 (흐린 이미지용)
        """
        if plate_image is None or plate_image.size == 0:
            return plate_image
        
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        
        # 크기 조정
        h, w = gray.shape
        if h < 150:
            scale = 150 / h
            gray = cv2.resize(gray, None, fx=scale, fy=scale, 
                            interpolation=cv2.INTER_CUBIC)
        
        # 강한 샤프닝
        kernel = np.array([
            [-1, -1, -1],
            [-1, 12, -1],
            [-1, -1, -1]
        ])
        sharpened = cv2.filter2D(gray, -1, kernel)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(sharpened)
        
        return enhanced

    def preprocess_binary(self, plate_image: np.ndarray) -> np.ndarray:
        """
        이진화 강화 전처리 (명암 문제용)
        """
        if plate_image is None or plate_image.size == 0:
            return plate_image
        
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        
        # 크기 조정
        h, w = gray.shape
        if h < 150:
            scale = 150 / h
            gray = cv2.resize(gray, None, fx=scale, fy=scale, 
                            interpolation=cv2.INTER_CUBIC)
        
        # 강한 이진화
        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # 모폴로지
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return morphed
    
    def extract_korean_plate(self, text: str) -> str:
        """
        텍스트에서 한국 번호판 형식 추출
        
        지원 형식:
        1. 일반: 12가3456, 123나4567
        2. 오토바이: 서울4파3151
        3. 영업용: 서울12가3456
        4. 전기차: 12하3456
        """
        # 오인식 보정 먼저
        text = self.correct_ocr_errors(text)
        
        # 패턴 우선순위대로 시도
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

    def correct_ocr_errors(self, text: str) -> str:
        """
        OCR 오인식 보정 (영어↔숫자, 한글 오인식)
        """
        # 영어 → 숫자 보정
        char_corrections = {
            'O': '0', 'o': '0',
            'I': '1', 'l': '1',
            'Z': '2',
            'S': '5',
            'B': '8',
            'g': '9',
        }
        
        result = ""
        for char in text:
            result += char_corrections.get(char, char)
        
        # ✅ 한글 오인식 보정 (단어 단위)
        # 자주 틀리는 지역명/한글 쌍
        word_corrections = {
            '서물': '서울',
            '운산': '인산',  # 드물지만
            '정북': '전북',
            '정남': '전남',
            '충복': '충북',
            '경기': '경기',
            '부산': '부산',
            # 필요시 추가
        }
        
        for wrong, correct in word_corrections.items():
            result = result.replace(wrong, correct)
        
        return result

    def train(self, *args, **kwargs):
        raise NotImplementedError("easyocr 자체 학습은 여기서 지원하지 않습니다.")
