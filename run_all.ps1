# run_all.ps1 — Execute the entire HAR pipeline end-to-end.
#
# Usage (from the code/ directory):
#   .\run_all.ps1
#
# Skips already-cached stages (data_loader, feature_engineering). Use
# -Force to rebuild everything.
#
param(
    [switch]$Force,
    [switch]$SkipCNN,            # skip train_cnn.py (slow on CPU) — handy for retries
    [switch]$SkipReport          # skip make_report_docx.py
)

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT

$PY = Join-Path $ROOT ".venv\Scripts\python.exe"
if (-not (Test-Path $PY)) {
    Write-Host "Creating .venv ..."
    python -m venv .venv
}

Write-Host "Installing requirements ..."
& $PY -m pip install --quiet -r requirements.txt
& $PY -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu

function Run-Step {
    param([string]$Script, [string]$ExpectFile, [string[]]$ScriptArgs = @())
    if ((-not $Force) -and $ExpectFile -and (Test-Path $ExpectFile)) {
        Write-Host "[SKIP] $Script (output $ExpectFile already exists)"
        return
    }
    Write-Host "`n=== Running $Script $ScriptArgs ===" -ForegroundColor Cyan
    & $PY $Script @ScriptArgs
    if ($LASTEXITCODE -ne 0) { throw "$Script failed with exit code $LASTEXITCODE" }
}

Run-Step "data_loader.py"           "outputs\cache\train_long.parquet"
Run-Step "eda.py"                   "outputs\figures\01_label_distribution.png"
Run-Step "naive_baseline.py"        "outputs\cache\naive_oof.npz"
Run-Step "feature_engineering.py"   "outputs\cache\feats_train_v4.parquet"
# Tree models — bag 3 seeds each (variance reduction).
Run-Step "train_lgbm.py"            "outputs\cache\lgbm_oof.npz"   @("--seeds","7,13,42")
Run-Step "train_xgb.py"             "outputs\cache\xgb_oof.npz"    @("--seeds","7,13,42")
Run-Step "train_catboost.py"        "outputs\cache\cat_oof.npz"
Run-Step "train_mlp.py"             "outputs\cache\mlp_oof.npz"
# ROCKET — diverse random-kernel model on the raw signal.
Run-Step "train_rocket.py"          "outputs\cache\rocket_oof.npz"
if (-not $SkipCNN) {
    Run-Step "train_cnn.py"         "outputs\cache\cnn_oof.npz"
}
# Ensemble needs every base model present; run once they're all cached.
Run-Step "ensemble.py"              "outputs\cache\ensemble_oof.npz"
Run-Step "make_report_assets.py"    "outputs\figures\07_confusion_primary.png"
if (-not $SkipReport) {
    Run-Step "make_report_docx.py"  ""   # always run; user wants fresh report
}

Write-Host "`nAll steps complete." -ForegroundColor Green
Write-Host "Final submission: outputs\submissions\submission_final.csv"
Write-Host "Word report:      outputs\report\DM_asg3_*.docx"
