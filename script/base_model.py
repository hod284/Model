from abc import ABC, abstractmethod


class BaseModel(ABC):
    """공통 인터페이스"""

    @abstractmethod
    def predict(self, *args, **kwargs):
        """추론"""
        # 패스는 여기서 안할거니까 상속받은 함수에 구현해주세요
        pass
# 아무인자나 넣으세요 *args, **kwargs
    def train(self, *args, **kwargs):
        """학습 (필요 없으면 override 안 해도 됨)"""
        raise NotImplementedError("Train is not implemented")

    def save(self, path: str):
        raise NotImplementedError("Save is not implemented")

    def load(self, path: str):
        raise NotImplementedError("Load is not implemented")