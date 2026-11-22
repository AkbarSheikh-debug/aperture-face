"""Assemble deploy/vercel/ - a self-contained, stateless, serverless deployment of the web app.

    python deploy/build_vercel.py          # (re)builds deploy/vercel from the sources in this repo

Everything the function needs is copied in (code, the INT8 ONNX model, the face detector, the results file,
the web page); nothing is imported from outside the folder, so `vercel` can deploy it as-is.
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deploy" / "vercel"

FILES = [
    "common/__init__.py", "common/align.py", "common/onnx_runtime.py",
    "api/__init__.py", "api/engine.py",
    "checkpoints/face_r18_int8.onnx", "checkpoints/face_detection_yunet_2023mar.onnx",
    "results/evaluation_results.json",
]
TREES = ["api/static"]


def main():
    if OUT.exists():
        for child in OUT.iterdir():
            if child.name not in {"app.py", "requirements.txt", "vercel.json", ".python-version", ".vercelignore", ".vercel"}:
                shutil.rmtree(child) if child.is_dir() else child.unlink()
    for rel in FILES:
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)
    for rel in TREES:
        shutil.copytree(ROOT / rel, OUT / rel, dirs_exist_ok=True)
    mb = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"built {OUT.relative_to(ROOT)}  ({mb:.1f} MB of project files)")


if __name__ == "__main__":
    main()
