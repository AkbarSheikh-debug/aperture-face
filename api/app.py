"""Face verification micro-service (FastAPI + ONNX Runtime; no PyTorch at inference time).

Endpoints
  GET  /health                         model, threshold, enrolled count
  POST /verify    (image_a, image_b)   -> cosine similarity + match decision
  POST /enroll    (name, image)        -> stores ONLY the 512-D embedding (no image is kept)
  POST /identify  (image, top_k)       -> nearest enrolled identities (open-set: "unknown" below threshold)
  DELETE /enroll/{name}                -> right to erasure

Run locally : uvicorn api.app:app --port 8000          (from submission/)
Docker      : docker build -f api/Dockerfile -t face-verify . && docker run -p 8000:8000 face-verify
Config (env): FACE_MODEL (default checkpoints/face_r18_int8.onnx), FACE_THRESHOLD (default: value in
              results/evaluation_results.json), FACE_DB (default api/embeddings.sqlite)
"""
import json
import os
import sqlite3
import threading
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from fastapi.staticfiles import StaticFiles

from api import engine
from common.align import FaceAligner
from common.onnx_runtime import OnnxEmbedder

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(os.environ.get("FACE_MODEL", ROOT / "checkpoints" / "face_r18_int8.onnx"))
YUNET = Path(os.environ.get("FACE_DETECTOR", ROOT / "checkpoints" / "face_detection_yunet_2023mar.onnx"))
DB = Path(os.environ.get("FACE_DB", ROOT / "api" / "embeddings.sqlite"))


def default_threshold():
    if "FACE_THRESHOLD" in os.environ:
        return float(os.environ["FACE_THRESHOLD"])
    res = ROOT / "results" / "evaluation_results.json"
    edge = json.loads(res.read_text()).get("edge_threshold") if res.exists() else None
    if edge:
        return float(edge["threshold"])
    return float(json.loads(res.read_text())["verification"]["final_threshold"]["threshold"]) if res.exists() else 0.35


app = FastAPI(title="ResNet Face Verification", version="1.0")
app.mount("/static", StaticFiles(directory=ROOT / "api" / "static"), name="static")
embedder = OnnxEmbedder(MODEL, "cpu", threads=int(os.environ.get("FACE_THREADS", 4)))
aligner = FaceAligner(YUNET)
THRESHOLD = default_threshold()
_lock = threading.Lock()
_db = sqlite3.connect(DB, check_same_thread=False)
_db.execute("CREATE TABLE IF NOT EXISTS gallery (name TEXT, emb BLOB)")


def _embed_upload(upload: UploadFile) -> np.ndarray:
    data = np.frombuffer(upload.file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, f"{upload.filename}: not a decodable image")
    faces = aligner.detect(img)
    if len(faces) == 0:
        raise HTTPException(422, f"{upload.filename}: no face detected")
    crop, _ = aligner.align(img, pick="largest")
    return embedder.embed_crops([crop], tta=True)[0]


def _gallery():
    rows = _db.execute("SELECT name, emb FROM gallery").fetchall()
    if not rows:
        return [], np.zeros((0, 512), np.float32)
    names = [r[0] for r in rows]
    embs = np.stack([np.frombuffer(r[1], np.float32) for r in rows])
    # one template per identity = mean of its enrolled embeddings
    uniq = sorted(set(names))
    tmpl = np.stack([embs[[i for i, n in enumerate(names) if n == u]].mean(0) for u in uniq])
    return uniq, tmpl / np.linalg.norm(tmpl, axis=1, keepdims=True)


@app.post("/analyze")
def analyze_frame(image: UploadFile = File(...)):
    """One round trip per camera frame: detection + head pose + 512-D embedding. Stores nothing.
    The web page keeps the enrolled embeddings in the browser and matches there, so the same page also
    works against the stateless serverless deployment."""
    img = engine.decode(image.file.read())
    if img is None:
        raise HTTPException(400, "not a decodable image")
    return engine.analyze(aligner, embedder, img)


@app.post("/pose")
def pose(image: UploadFile = File(...)):
    """Detection + head-pose cues only (no embedding)."""
    img = engine.decode(image.file.read())
    if img is None:
        raise HTTPException(400, "not a decodable image")
    return engine.analyze(aligner, embedder, img, want_embedding=False)


@app.get("/about")
def about():
    return engine.about(ROOT / "results" / "evaluation_results.json")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "api" / "static" / "index.html")


@app.get("/gallery")
def gallery():
    names = [r[0] for r in _db.execute("SELECT name FROM gallery").fetchall()]
    return {"identities": [{"name": n, "templates": names.count(n)} for n in sorted(set(names))]}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL.name, "providers": embedder.providers, "threshold": THRESHOLD,
            "enrolled_identities": len(_gallery()[0])}


@app.post("/verify")
def verify(image_a: UploadFile = File(...), image_b: UploadFile = File(...)):
    a, b = _embed_upload(image_a), _embed_upload(image_b)
    sim = float(a @ b)
    return {"cosine_similarity": round(sim, 4), "threshold": THRESHOLD, "match": sim >= THRESHOLD}


@app.post("/enroll")
def enroll(name: str = Form(...), image: UploadFile = File(...)):
    e = _embed_upload(image).astype(np.float32)
    with _lock:
        _db.execute("INSERT INTO gallery VALUES (?, ?)", (name, e.tobytes()))
        _db.commit()
    return {"enrolled": name, "stored": "512-D embedding only"}


@app.delete("/enroll/{name}")
def forget(name: str):
    with _lock:
        n = _db.execute("DELETE FROM gallery WHERE name = ?", (name,)).rowcount
        _db.commit()
    return {"deleted_embeddings": n}


@app.post("/identify")
def identify(image: UploadFile = File(...), top_k: int = Form(5)):
    names, tmpl = _gallery()
    if not names:
        raise HTTPException(409, "gallery is empty - enroll someone first")
    sims = tmpl @ _embed_upload(image)
    order = np.argsort(-sims)[:top_k]
    best = float(sims[order[0]])
    return {"identity": names[order[0]] if best >= THRESHOLD else "unknown", "score": round(best, 4),
            "threshold": THRESHOLD,
            "candidates": [{"name": names[i], "cosine_similarity": round(float(sims[i]), 4)} for i in order]}
