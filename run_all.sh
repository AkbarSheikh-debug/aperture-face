#!/usr/bin/env bash
# End-to-end reproduction. Run from submission/ with the venv active.
#   bash run_all.sh            full run (~2.5 h on one RTX 5080, plus ~15 GB of downloads)
#   bash run_all.sh --quick    smoke test on a tiny subset (minutes) - writes to the same folders
set -euo pipefail
Q=""; [[ "${1:-}" == "--quick" ]] && Q="--quick"
PY=${PYTHON:-python}
DATA=${FACE_DATA:-../data}

mkdir -p "$DATA/raw/digiface"
# ---- downloads (skipped if present)
[ -d "$DATA/raw/lfw" ] || { curl -L -o "$DATA/raw/lfw.tgz" https://ndownloader.figshare.com/files/5976018; tar -xzf "$DATA/raw/lfw.tgz" -C "$DATA/raw"; }
[ -f "$DATA/raw/face_detection_yunet_2023mar.onnx" ] || curl -L -o "$DATA/raw/face_detection_yunet_2023mar.onnx" \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
for p in 0-1999 2000-3999 4000-5999 6000-7999 8000-9999; do
  [ -f "$DATA/raw/digiface/subjects_${p}_72_imgs.zip" ] || curl -L -o "$DATA/raw/digiface/subjects_${p}_72_imgs.zip" \
    "https://facesyntheticspubwedata.z6.web.core.windows.net/wacv-2023/subjects_${p}_72_imgs.zip"
done
[ -d "$DATA/raw/digiface/extracted" ] || for f in "$DATA"/raw/digiface/*.zip; do unzip -q -o "$f" -d "$DATA/raw/digiface/extracted"; done

# ---- Task 1-3
$PY dataset_preparation.py $Q
$PY generate_pairs.py
# ---- Task 2 (two-stage training)
$PY train.py --config configs/pretrain_digiface.yaml $Q
$PY train.py --config configs/finetune_lfw.yaml $Q
# ---- Task 4-7
$PY evaluate_pairs.py
$PY roc_analysis.py --operating-point far_1e-3
$PY gallery_probe.py
# ---- ablation + analysis
$PY -m analysis.ablation
$PY -m analysis.error_analysis
$PY -m analysis.gradcam
# ---- edge track
$PY -m edge.distill $Q
$PY -m edge.export_onnx
$PY -m edge.quantize_int8
$PY -m edge.build_tensorrt
$PY -m edge.benchmark_edge
# ---- tests + report
$PY -m pytest tests -q
$PY -m analysis.make_report
