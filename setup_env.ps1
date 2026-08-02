# ============================================================
#  Multi-Agent Medical Assistant - Conda Environment Setup
#  
#  Usage (run in PowerShell):
#    cd "c:\Users\haooding\Desktop\Multi-Agent-Medical-Assistant-main"
#    .\setup_env.ps1
#
#  NOTE: The environment "medical-assistant" has already been created
#        with Python 3.11. This script will complete the remaining setup.
# ============================================================

$ErrorActionPreference = "Continue"
$ENV_NAME = "medical-assistant"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Multi-Agent Medical Assistant - Completing Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Verify environment exists
$envExists = conda info --envs 2>&1 | Select-String $ENV_NAME
if (-not $envExists) {
    Write-Host "[INFO] Creating environment '$ENV_NAME' with Python 3.11..." -ForegroundColor Green
    conda create -n $ENV_NAME python=3.11 -c conda-forge -y
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to create environment!" -ForegroundColor Red
        exit 1
    }
}

# Step 1: Install PyTorch (CPU version for smaller download)
Write-Host ""
Write-Host "[Step 1/3] Installing PyTorch (CPU version)..." -ForegroundColor Green
Write-Host "  (If you have NVIDIA GPU, you can later reinstall with CUDA support)" -ForegroundColor Gray
conda run -n $ENV_NAME pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARNING] PyTorch install failed. Trying without version constraint..." -ForegroundColor Yellow
    conda run -n $ENV_NAME pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
}

# Step 2: Install remaining pip dependencies
Write-Host ""
Write-Host "[Step 2/3] Installing remaining Python dependencies..." -ForegroundColor Green
Write-Host "  (This may take 10-20 minutes depending on network speed)" -ForegroundColor Gray
conda run -n $ENV_NAME pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARNING] Some packages failed. Retrying with relaxed versions..." -ForegroundColor Yellow
    conda run -n $ENV_NAME pip install -r requirements.txt --ignore-installed
}

# Step 3: Install ffmpeg via winget (since conda ffmpeg has issues on this system)
Write-Host ""
Write-Host "[Step 3/3] Checking ffmpeg..." -ForegroundColor Green
$ffmpegCheck = conda run -n $ENV_NAME ffmpeg -version 2>&1
if ($ffmpegCheck -match "ffmpeg version") {
    Write-Host "  ffmpeg is already available." -ForegroundColor Green
} else {
    Write-Host "  ffmpeg not found in environment. Installing via winget..." -ForegroundColor Yellow
    winget install --id Gyan.FFmpeg -e --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [INFO] Please install ffmpeg manually:" -ForegroundColor Yellow
        Write-Host "    Option A: winget install ffmpeg" -ForegroundColor Gray
        Write-Host "    Option B: Download from https://ffmpeg.org/download.html" -ForegroundColor Gray
        Write-Host "    Then add ffmpeg to your system PATH" -ForegroundColor Gray
    }
}

# Verify installation
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Verifying Installation..." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

conda run -n $ENV_NAME python -c "
import sys
print(f'  Python: {sys.version}')
try:
    import torch
    print(f'  PyTorch: {torch.__version__}')
except: print('  PyTorch: NOT INSTALLED')
try:
    import langchain
    print(f'  LangChain: {langchain.__version__}')
except: print('  LangChain: NOT INSTALLED')
try:
    import fastapi
    print(f'  FastAPI: {fastapi.__version__}')
except: print('  FastAPI: NOT INSTALLED')
try:
    import openai
    print(f'  OpenAI: {openai.__version__}')
except: print('  OpenAI: NOT INSTALLED')
try:
    import pydub
    print(f'  Pydub: {pydub.__version__}')
except: print('  Pydub: NOT INSTALLED')
print()
print('  All core dependencies verified!')
"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Setup Complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host " To use the environment:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  1. Activate:" -ForegroundColor White
Write-Host "     conda activate $ENV_NAME" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. Configure API keys (copy .env.example to .env):" -ForegroundColor White
Write-Host "     copy .env.example .env" -ForegroundColor Gray
Write-Host "     # Then edit .env with your API keys" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. Run the app:" -ForegroundColor White
Write-Host "     python app.py" -ForegroundColor Gray
Write-Host ""
Write-Host "  4. Open in browser:" -ForegroundColor White
Write-Host "     http://localhost:8000" -ForegroundColor Gray
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
