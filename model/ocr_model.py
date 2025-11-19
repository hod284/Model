import numpy as np
import cv2
import  easyocr

from base_model import BaseModel

class OcrModel(BaseModel):

    def __init__(self,  use_dummy: bool =False, language =None):

           self.use_dummy = use_dummy

           if language is None:
                languages = ['ko', 'en']

           if not use_dummy:
              self.reader = easyocr.Reader(languages, False)
           else:
             self.reader = None
      # -> 이거는 주석용 그러니까 컴파일러에 영향없음
    def preprocess(self, plate_image: np.ndarray) -> np.ndarray:
         if plate_image is None or plate_image.size ==0:
               return plate_image
         gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
         return gray

    def predic (self, plate_image: np.ndarray) -> str:
        if self.use_dummy:
          return "12가3456"
        if self.reader is None:
            raise RuntimeError("easyocr render가 초기화 되지 않았습니다.")
        if plate_image is None or plate_image.size ==0:
            return ""
        processed = self.preprocess(plate_image)
        results = self.reader.readtext(processed, detail = 0)
        if not results:
            return ""
        best = max(results, key =len)
        best = best.replace(" ","")
        return best


    def train(self, *args, **kwargs):
       raise NotImplementedError("easyocr 자체 학습은 여기서 지원하지 않습니다.")
