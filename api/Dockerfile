# Minimal CPU inference image: numpy + opencv-headless + onnxruntime (no PyTorch, no CUDA).
# Build from submission/:  docker build -f api/Dockerfile -t face-verify .
# Multi-arch ready: the same file builds for linux/arm64 (Raspberry Pi 5 / Jetson CPU) with
#   docker buildx build --platform linux/arm64 -f api/Dockerfile -t face-verify:arm64 .
FROM python:3.12-slim

LABEL org.opencontainers.image.title="Aperture face recognition"
LABEL org.opencontainers.image.description="ResNet face verification and identification: web app and API, CPU only"
LABEL org.opencontainers.image.source="https://github.com/AkbarSheikh-debug/aperture-face"

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 FACE_THREADS=4
WORKDIR /app

RUN pip install --no-cache-dir numpy==2.5.4 opencv-python-headless==5.1.0.95 onnxruntime==1.32.0 \
        fastapi==0.165.0 uvicorn==0.54.0 python-multipart==0.0.34

COPY common/__init__.py common/align.py common/onnx_runtime.py ./common/
COPY api/app.py api/engine.py ./api/
COPY api/static ./api/static
COPY checkpoints/face_r18_int8.onnx checkpoints/face_detection_yunet_2023mar.onnx ./checkpoints/
COPY results/evaluation_results.json ./results/evaluation_results.json
RUN touch api/__init__.py

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
