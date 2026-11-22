"""Stateless face-recognition service for serverless hosting (Vercel).

No database and no files are written: each request is analysed in memory and discarded. The web page keeps
the enrolled embeddings in the visitor's own browser and does the matching there, so one deployment can serve
many people without their faces ever being shared or stored on the server.

  GET  /            the web app
  GET  /health      model, runtime, match threshold
  GET  /about       evaluation numbers shown on the About tab
  POST /analyze     image -> face box, head pose, 512-D embedding
  POST /verify      two images -> cosine similarity + decision (convenience API)
"""
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api import engine
from common.align import FaceAligner
from common.onnx_runtime import OnnxEmbedder

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "evaluation_results.json"
MODEL = ROOT / "checkpoints" / "face_r18_int8.onnx"
MAX_BYTES = 4_400_000   # Vercel rejects bodies over ~4.5 MB anyway; fail with a clear message instead

app = FastAPI(title="ResNet Face Recognition (stateless)", version="1.0")
app.mount("/static", StaticFiles(directory=ROOT / "api" / "static"), name="static")
embedder = OnnxEmbedder(MODEL, "cpu", threads=int(os.environ.get("FACE_THREADS", 2)))
aligner = FaceAligner(ROOT / "checkpoints" / "face_detection_yunet_2023mar.onnx")
THRESHOLD = float(os.environ.get("FACE_THRESHOLD", engine.default_threshold(RESULTS)))


def _read(upload: UploadFile):
    data = upload.file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image too large (max ~4 MB) - resize it first")
    img = engine.decode(data)
    if img is None:
        raise HTTPException(400, "not a decodable image")
    return img


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "api" / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL.name, "providers": embedder.providers, "threshold": THRESHOLD,
            "mode": "stateless"}


@app.get("/about")
def about():
    return engine.about(RESULTS)


@app.post("/analyze")
def analyze(image: UploadFile = File(...)):
    return engine.analyze(aligner, embedder, _read(image))


@app.post("/verify")
def verify(image_a: UploadFile = File(...), image_b: UploadFile = File(...)):
    a, b = (engine.analyze(aligner, embedder, _read(u)) for u in (image_a, image_b))
    for tag, r in (("image_a", a), ("image_b", b)):
        if "embedding" not in r:
            raise HTTPException(422, f"{tag}: " + ("no face detected" if not r["face"] else "more than one face"))
    sim = sum(x * y for x, y in zip(a["embedding"], b["embedding"]))
    return {"cosine_similarity": round(sim, 4), "threshold": THRESHOLD, "match": sim >= THRESHOLD}
