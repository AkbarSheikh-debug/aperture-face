# End-to-end reproduction on Windows. Run from submission\ :
#   .\run_all.ps1            full run (~2.5 h on one RTX 5080, plus ~15 GB of downloads)
#   .\run_all.ps1 -Quick     smoke test on a tiny subset
param([switch]$Quick)
$ErrorActionPreference = "Stop"
$Q = if ($Quick) { "--quick" } else { "" }
$PY = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$DATA = if ($env:FACE_DATA) { $env:FACE_DATA } else { "..\data" }
$RAW = Join-Path $DATA "raw"
New-Item -ItemType Directory -Force (Join-Path $RAW "digiface") | Out-Null

function Invoke-Step([string[]]$cmd) {
    Write-Host ">> $($cmd -join ' ')" -ForegroundColor Cyan
    & $PY @cmd
    if ($LASTEXITCODE -ne 0) { throw "step failed: $($cmd -join ' ')" }
}

# ---- downloads (skipped if present)
if (-not (Test-Path (Join-Path $RAW "lfw"))) {
    curl.exe -L -o (Join-Path $RAW "lfw.tgz") https://ndownloader.figshare.com/files/5976018
    tar -xzf (Join-Path $RAW "lfw.tgz") -C $RAW
}
$yunet = Join-Path $RAW "face_detection_yunet_2023mar.onnx"
if (-not (Test-Path $yunet)) {
    curl.exe -L -o $yunet https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
}
foreach ($p in "0-1999", "2000-3999", "4000-5999", "6000-7999", "8000-9999") {
    $zip = Join-Path $RAW "digiface\subjects_${p}_72_imgs.zip"
    if (-not (Test-Path $zip)) {
        curl.exe -L -o $zip "https://facesyntheticspubwedata.z6.web.core.windows.net/wacv-2023/subjects_${p}_72_imgs.zip"
    }
}
$ext = Join-Path $RAW "digiface\extracted"
if (-not (Test-Path $ext)) {
    Get-ChildItem (Join-Path $RAW "digiface\*.zip") | ForEach-Object { tar -xf $_.FullName -C (New-Item -ItemType Directory -Force $ext) }
}

$steps = @(
    @("dataset_preparation.py", $Q), @("generate_pairs.py"),
    @("train.py", "--config", "configs/pretrain_digiface.yaml", $Q),
    @("train.py", "--config", "configs/finetune_lfw.yaml", $Q),
    @("evaluate_pairs.py"), @("roc_analysis.py", "--operating-point", "far_1e-3"), @("gallery_probe.py"),
    @("-m", "analysis.ablation"), @("-m", "analysis.error_analysis"), @("-m", "analysis.gradcam"),
    @("-m", "edge.distill", $Q), @("-m", "edge.export_onnx"), @("-m", "edge.quantize_int8"),
    @("-m", "edge.build_tensorrt"), @("-m", "edge.benchmark_edge"),
    @("-m", "pytest", "tests", "-q"), @("-m", "analysis.make_report")
)
foreach ($s in $steps) { Invoke-Step ($s | Where-Object { $_ -ne "" }) }
