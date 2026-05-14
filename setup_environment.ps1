# =============================================================================
# Project Overdrive — Full Environment Setup
# =============================================================================
# One-time setup script for the heterogeneous compute stack.
# Run this once after cloning the repository.

param(
    [switch]$SkipModels,
    [switch]$SkipRAG,
    [switch]$SkipNPU
)

Write-Host "============================================" -ForegroundColor Magenta
Write-Host "  Project Overdrive - Environment Setup" -ForegroundColor Magenta
Write-Host "============================================" -ForegroundColor Magenta

$ProjectRoot = $PSScriptRoot

# ── Early Dependency Check ──────────────────────────────────────────────────
$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmake) {
    # Try default installation path as fallback
    $defaultPath = "C:\Program Files\CMake\bin\cmake.exe"
    if (Test-Path $defaultPath) {
        $cmake = $defaultPath
        # Temporarily add to path for this session
        $env:PATH = "C:\Program Files\CMake\bin;" + $env:PATH
        Write-Host "Found CMake in default path: $defaultPath" -ForegroundColor Green
    } else {
        Write-Host "ERROR: CMake not found. It is required for building llama-cpp-python with CUDA." -ForegroundColor Red
        Write-Host "Please download and install CMake from:" -ForegroundColor Yellow
        Write-Host "  https://github.com/Kitware/CMake/releases/download/v4.3.2/cmake-4.3.2-windows-x86_64.msi" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "IMPORTANT: If you JUST installed it, you MUST RESTART your terminal (PowerShell) for the changes to take effect." -ForegroundColor Yellow
        exit 1
    }
}

# ── Step 1: Python venv ──────────────────────────────────────────────────────
Write-Host ""
Write-Host "[1/6] Checking Python virtual environment..." -ForegroundColor Cyan

if (-not (Test-Path "$ProjectRoot\venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv "$ProjectRoot\venv"
}

# Upgrade pip and core tools
Write-Host "Upgrading pip and core tools..." -ForegroundColor Yellow
& "$ProjectRoot\venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel --quiet

# Activate and install base dependencies
Write-Host "Installing Python dependencies (this may take a moment)..." -ForegroundColor Yellow
& "$ProjectRoot\venv\Scripts\pip.exe" install -r "$ProjectRoot\requirements.txt"

# Verify critical dependencies
Write-Host "Verifying installations..." -ForegroundColor Cyan
$deps = @("chromadb", "llama_cpp", "openvino", "langchain", "langgraph")
foreach ($dep in $deps) {
    $check = & "$ProjectRoot\venv\Scripts\python.exe" -c "try: import $dep.replace('_', '-'); print('ok'); except ImportError: print('missing')" 2>$null
    if ($check -eq 'missing') {
        # Try without replacement for some modules
        $check = & "$ProjectRoot\venv\Scripts\python.exe" -c "try: import $dep; print('ok'); except ImportError: print('missing')" 2>$null
    }
    if ($check -eq 'missing') {
        Write-Host "  [!] Warning: $dep is missing or failed to install." -ForegroundColor Yellow
    } else {
        Write-Host "  [ok] $dep" -ForegroundColor Green
    }
}

# ── Step 2: llama-cpp-python with CUDA ───────────────────────────────────────
Write-Host ""
Write-Host "[2/6] Setting up llama-cpp-python with CUDA..." -ForegroundColor Cyan

$llamaCppInstalled = & "$ProjectRoot\venv\Scripts\python.exe" -c "
try:
    import llama_cpp; print('yes')
except ImportError:
    print('no')
" 2>$null

if ($llamaCppInstalled -eq 'no') {
    Write-Host "Running CUDA setup..." -ForegroundColor Yellow
    & "$ProjectRoot\setup_cuda.ps1"
} else {
    Write-Host "llama-cpp-python already installed." -ForegroundColor Green
}

# ── Step 3: Download GPU model (GGUF) ───────────────────────────────────────
Write-Host ""
Write-Host "[3/6] Checking GPU model files..." -ForegroundColor Cyan

$gpuModelDir = "$ProjectRoot\models\gpu"
if (-not (Test-Path $gpuModelDir)) {
    New-Item -ItemType Directory -Path $gpuModelDir -Force | Out-Null
}

$ggufFile = "$gpuModelDir\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
if ($SkipModels) {
    Write-Host "Skipping model download (--SkipModels)." -ForegroundColor Yellow
} elseif (Test-Path $ggufFile) {
    Write-Host "GPU model already exists: $ggufFile" -ForegroundColor Green
} else {
    Write-Host "Downloading Meta-Llama-3.1-8B-Instruct Q4_K_M GGUF..." -ForegroundColor Yellow
    Write-Host "This is a ~4.9 GB download. Please be patient." -ForegroundColor Yellow
    
    $ggufUrl = "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    
    try {
        # Use BITS transfer for resumable download
        Start-BitsTransfer -Source $ggufUrl -Destination $ggufFile -DisplayName "Downloading GGUF model"
        Write-Host "GPU model downloaded successfully." -ForegroundColor Green
    } catch {
        Write-Host "BITS transfer failed. Trying curl..." -ForegroundColor Yellow
        curl -L -o $ggufFile $ggufUrl
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: Failed to download model. Download manually from:" -ForegroundColor Red
            Write-Host "  $ggufUrl" -ForegroundColor Yellow
            Write-Host "  Save to: $ggufFile" -ForegroundColor Yellow
        }
    }
}

# ── Step 4: NPU model (OpenVINO IR) ─────────────────────────────────────────
Write-Host ""
Write-Host "[4/6] Checking NPU model files..." -ForegroundColor Cyan

$npuModelDir = "$ProjectRoot\models\npu"
if (-not (Test-Path $npuModelDir)) {
    New-Item -ItemType Directory -Path $npuModelDir -Force | Out-Null
}

if ($SkipNPU) {
    Write-Host "Skipping NPU setup (--SkipNPU)." -ForegroundColor Yellow
} elseif (Test-Path "$npuModelDir\openvino_model.xml") {
    Write-Host "NPU model already exists." -ForegroundColor Green
} else {
    Write-Host "Converting model to OpenVINO IR (INT4)..." -ForegroundColor Yellow
    Write-Host "This may take several minutes and requires ~16GB RAM." -ForegroundColor Yellow
    
    & "$ProjectRoot\venv\Scripts\python.exe" -c @"
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer
import os

model_id = 'Qwen/Qwen2.5-7B-Instruct'
output_dir = os.path.join('$($npuModelDir.Replace('\','/'))')

print(f'Exporting {model_id} to OpenVINO IR (INT4)...')
try:
    model = OVModelForCausalLM.from_pretrained(
        model_id,
        export=True,
        load_in_4bit=True,
    )
    model.save_pretrained(output_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(output_dir)
    
    print('NPU model export complete.')
except Exception as e:
    print(f'NPU export failed: {e}')
    print('You can run this manually later or use CPU fallback.')
"@
}

# ── Step 5: Knowledge base data ──────────────────────────────────────────────
Write-Host ""
Write-Host "[5/6] Setting up RAG knowledge base..." -ForegroundColor Cyan

if ($SkipRAG) {
    Write-Host "Skipping RAG setup (--SkipRAG)." -ForegroundColor Yellow
} else {
    # Fetch knowledge base data
    & "$ProjectRoot\scripts\fetch_knowledge.ps1"
    
    # Run ingestion
    Write-Host "Ingesting knowledge base into ChromaDB..." -ForegroundColor Yellow
    & "$ProjectRoot\venv\Scripts\python.exe" -m rag.ingest --source "$ProjectRoot\data\knowledge" --persist-dir "$ProjectRoot\data\chromadb"
}

# ── Step 6: Create data directories ─────────────────────────────────────────
Write-Host ""
Write-Host "[6/6] Creating data directories..." -ForegroundColor Cyan

$dirs = @(
    "$ProjectRoot\data\chromadb",
    "$ProjectRoot\data\knowledge\gtfobins",
    "$ProjectRoot\data\knowledge\payloads",
    "$ProjectRoot\data\knowledge\exploitdb",
    "$ProjectRoot\data\knowledge\cheatsheets",
    "$ProjectRoot\models\gpu",
    "$ProjectRoot\models\npu"
)
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Write-Host "All directories created." -ForegroundColor Green

# ── Summary ──────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Ensure GGUF model exists at: models\gpu\" -ForegroundColor White
Write-Host "  2. Run: .\run_prototype.ps1" -ForegroundColor White
Write-Host "  3. Or run directly: python agents\react_loop.py --question '...'" -ForegroundColor White
