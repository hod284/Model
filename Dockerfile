# Python 기반 이미지
FROM python:3.10-slim

# 파이썬 기본 설정
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 작업 디렉터리
WORKDIR /app

# EasyOCR / OpenCV에 필요한 시스템 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
       libgl1 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# 파이썬 패키지 설치
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사 (server.py, proto에서 생성된 *_pb2.py 등 포함)
COPY . /app

# gRPC 포트
EXPOSE 50051

# gRPC 서버 실행 (엔트리 포인트)
CMD ["python", "server.py"]