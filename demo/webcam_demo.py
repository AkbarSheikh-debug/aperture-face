"""Live webcam demo (ONNX Runtime, CPU) - enrol yourself, then verify / identify in real time.

  python -m demo.webcam_demo --enroll "Akbar"      # look at the camera; captures 10 aligned frames
  python -m demo.webcam_demo                       # live identification with FPS overlay (q to quit)
  python -m demo.webcam_demo --camera clip.mp4 --headless   # same, from a video file, no window (for testing)

Gallery is stored as embeddings only in demo/gallery.npz.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from common.align import FaceAligner
from common.onnx_runtime import OnnxEmbedder

ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "demo" / "gallery.npz"


def load_gallery():
    if not GALLERY.exists():
        return {}
    z = np.load(GALLERY, allow_pickle=True)
    return dict(zip(z["names"], z["embs"]))


def save_gallery(g):
    np.savez(GALLERY, names=np.array(list(g)), embs=np.stack(list(g.values())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="0", help="camera index, or a video file path")
    ap.add_argument("--headless", action="store_true", help="no window; print labels instead")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--model", default=str(ROOT / "checkpoints" / "face_r18_int8.onnx"))
    ap.add_argument("--enroll", default=None, help="name to enrol")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()
    res = ROOT / "results" / "evaluation_results.json"
    thr = args.threshold
    if thr is None and res.exists():
        r = json.loads(res.read_text())
        thr = (r.get("edge_threshold") or r["verification"]["final_threshold"])["threshold"]
    thr = thr if thr is not None else 0.35
    aligner = FaceAligner(ROOT / "checkpoints" / "face_detection_yunet_2023mar.onnx")
    emb = OnnxEmbedder(args.model, "cpu", threads=4)
    cap = cv2.VideoCapture(int(args.camera) if args.camera.isdigit() else args.camera)
    n_frames = 0
    gallery = load_gallery()
    captured, t_prev, fps = [], time.time(), 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = aligner.align_all(frame)
        if faces:
            embs = emb.embed_crops([c for c, _, _ in faces])
            for (crop, (x, y, w, h), score), e in zip(faces, embs):
                if args.enroll:
                    label, color = f"enrolling {len(captured)}/{args.frames}", (0, 200, 255)
                    if len(faces) == 1:
                        captured.append(e)
                elif gallery:
                    names = list(gallery)
                    sims = np.stack(list(gallery.values())) @ e
                    k = int(np.argmax(sims))
                    match = sims[k] >= thr
                    label = f"{names[k] if match else 'unknown'}  {sims[k]:.2f}"
                    color = (60, 200, 60) if match else (60, 60, 230)
                else:
                    label, color = "gallery empty - run with --enroll NAME", (200, 200, 200)
                if args.headless:
                    print(f"frame {n_frames}: {label}", flush=True)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(1e-6, now - t_prev)
        t_prev = now
        cv2.putText(frame, f"{fps:.1f} FPS  thr={thr:.2f}  {Path(args.model).name}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if not args.headless:
            cv2.imshow("Face verification (q to quit)", frame)
        n_frames += 1
        if args.enroll and len(captured) >= args.frames:
            t = np.mean(captured, 0)
            gallery[args.enroll] = t / np.linalg.norm(t)
            save_gallery(gallery)
            print(f"enrolled {args.enroll} from {len(captured)} frames")
            args.enroll, captured = None, []
        if args.max_frames and n_frames >= args.max_frames:
            break
        if not args.headless and cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    if not args.headless:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
