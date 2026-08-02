# ============================================================
#  Multi-Agent Medical Assistant - Install Dependencies
#  
#  The conda environment "medical-assistant" (Python 3.11 + ffmpeg)
#  has already been created. Run this script to install all pip packages.
#
#  Usage:
#    cd "c:\Users\haooding\Desktop\Multi-Agent-Medical-Assistant-main"
#    .\install_deps.ps1
# ============================================================

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Installing all pip dependencies..." -ForegroundColor Cyan
Write-Host " This will take approximately 15-30 minutes." -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$pythonExe = "D:\code\miniforge\envs\medical-assistant\python.exe"

# Verify Python is accessible
if (-not (Test-Path $pythonExe)) {
    Write-Host "[ERROR] Python not found at: $pythonExe" -ForegroundColor Red
    Write-Host "Please re-create the environment:" -ForegroundColor Yellow
    Write-Host "  conda create -n medical-assistant python=3.11 -c conda-forge -y" -ForegroundColor Gray
    exit 1
}

# Step 1: Install PyTorch CPU first
Write-Host "[1/2] Installing PyTorch (CPU version)..." -ForegroundColor Green
& $pythonExe -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu
Write-Host ""

# Step 2: Install all remaining packages
Write-Host "[2/2] Installing remaining dependencies from requirements..." -ForegroundColor Green
& $pythonExe -m pip install -r requirements_no_pip.txt
Write-Host ""

# Verify key packages
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Verifying installation..." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
& $pythonExe -c @"
import sys
print(f'  Python: {sys.version.split()[0]}')
packages = {
    'torch': 'PyTorch',
    'langchain': 'LangChain', 
    'fastapi': 'FastAPI',
    'openai': 'OpenAI',
    'langgraph': 'LangGraph',
    'qdrant_client': 'Qdrant',
    'pydub': 'Pydub',
    'elevenlabs': 'ElevenLabs',
    'docling': 'Docling',
    'easyocr': 'EasyOCR',
}
for pkg, name in packages.items():
    try:
        m = __import__(pkg)
        ver = getattr(m, '__version__', 'OK')
        print(f'  {name}: {ver} ✓')
    except ImportError:
        print(f'  {name}: NOT INSTALLED ✗')
"@

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " Done! To run the application:" -ForegroundColor Green
Write-Host "" -ForegroundColor White
Write-Host "  conda activate medical-assistant" -ForegroundColor Gray
Write-Host "  copy .env.example .env   # Then edit with your API keys" -ForegroundColor Gray
Write-Host "  python app.py" -ForegroundColor Gray
Write-Host "  # Open http://localhost:8000" -ForegroundColor Gray
Write-Host "============================================================" -ForegroundColor Green
