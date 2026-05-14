# =============================================================================
# Project Overdrive Startup Script — Heterogeneous Compute
# =============================================================================
# Starts the full stack: validates models, initializes RAG, launches UI.
# Primary backend: llama-cpp-python (CUDA) + OpenVINO (NPU) + ChromaDB (CPU)
# Ollama is available as a fallback with --UseOllama flag.

param(
    [switch]$UseOllama,
    [switch]$SkipRAG,
    [switch]$NoCuda
)

Write-Host "============================================" -ForegroundColor Magenta
Write-Host "  Project Overdrive - Heterogeneous CTF AI" -ForegroundColor Magenta
Write-Host "============================================" -ForegroundColor Magenta

$ProjectRoot = $PSScriptRoot

# Load environment from .env
if (Test-Path "$ProjectRoot\.env") {
    Write-Host "Loading configuration from .env..." -ForegroundColor Cyan
    Get-Content "$ProjectRoot\.env" | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            $name = $matches[1].Trim()
            $value = $matches[2].Split('#')[0].Trim().Trim('"').Trim("'")
            [System.Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
} else {
    Write-Host "No .env file found. Copy .env.example to .env to customize." -ForegroundColor Yellow
}

# Install Python dependencies
Write-Host ""
Write-Host "Installing Python dependencies..." -ForegroundColor Cyan
& "$ProjectRoot\venv\Scripts\python.exe" -m pip install -r "$ProjectRoot\requirements.txt" --quiet 2>$null

# ── Compute Backend Selection ────────────────────────────────────────────────
if ($UseOllama) {
    Write-Host ""
    Write-Host "Using Ollama backend (--UseOllama flag set)..." -ForegroundColor Yellow
    $env:LLM_PROVIDER = "ollama"
    $env:USE_LOCAL_LLM = "true"
    if (-not $env:LOCAL_MODEL_NAME) { $env:LOCAL_MODEL_NAME = "gemma4:e4b" }
    if (-not $env:LOCAL_API_BASE) { $env:LOCAL_API_BASE = "http://localhost:11434" }

    # Start Ollama container
    Write-Host "Starting Dockerized Ollama service..." -ForegroundColor Cyan
    try {
        docker-compose --profile ollama up -d ollama
        Write-Host "Pulling $env:LOCAL_MODEL_NAME model..." -ForegroundColor Cyan
        docker-compose --profile ollama exec -T ollama ollama pull $env:LOCAL_MODEL_NAME
    } catch {
        Write-Warning "Failed to start Ollama. Is Docker Desktop running?"
    }
} else {
    $env:LLM_PROVIDER = "llamacpp"

    # Validate GPU model
    $gpuModel = $env:GPU_MODEL_PATH
    if (-not $gpuModel) { $gpuModel = "models\gpu\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" }

    $gpuModelFull = Join-Path $ProjectRoot $gpuModel
    if (Test-Path $gpuModelFull) {
        $sizeMB = [math]::Round((Get-Item $gpuModelFull).Length / 1MB, 1)
        Write-Host "GPU model found: $gpuModel ($sizeMB MB)" -ForegroundColor Green
    } else {
        Write-Host "GPU model NOT found at: $gpuModel" -ForegroundColor Red
        Write-Host "Run setup_environment.ps1 to download, or use --UseOllama" -ForegroundColor Yellow
        exit 1
    }

    # Check NPU model (optional)
    $npuModel = $env:NPU_MODEL_PATH
    if (-not $npuModel) { $npuModel = "models\npu\" }
    $npuModelFull = Join-Path $ProjectRoot $npuModel
    if (Test-Path "$npuModelFull\openvino_model.xml") {
        Write-Host "NPU model found: $npuModel" -ForegroundColor Green
    } else {
        Write-Host "NPU model not found. NPU parsing will fall back to CPU." -ForegroundColor Yellow
    }
}

# ── RAG Knowledge Base ──────────────────────────────────────────────────────
if (-not $SkipRAG -and $env:RAG_ENABLED -ne "false") {
    $ragPersistDirSuffix = if ($env:RAG_PERSIST_DIR) { $env:RAG_PERSIST_DIR } else { "data\chromadb" }
    $ragDir = Join-Path $ProjectRoot $ragPersistDirSuffix
    if (Test-Path $ragDir) {
        Write-Host "RAG knowledge base found at: $ragDir" -ForegroundColor Green
    } else {
        Write-Host "RAG knowledge base not initialized. Running ingestion..." -ForegroundColor Yellow
        $knowledgeDirSuffix = if ($env:RAG_DATA_DIR) { $env:RAG_DATA_DIR } else { "data\knowledge" }
        $knowledgeDir = Join-Path $ProjectRoot $knowledgeDirSuffix
        if (Test-Path $knowledgeDir) {
            & "$ProjectRoot\venv\Scripts\python.exe" -m rag.ingest `
                --source $knowledgeDir --persist-dir $ragDir
        } else {
            Write-Host "No knowledge data found. Run scripts\fetch_knowledge.ps1 first." -ForegroundColor Yellow
        }
    }
}

# ── Summary ──────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Green
Write-Host "  Provider:  $env:LLM_PROVIDER"
if ($env:LLM_PROVIDER -eq "llamacpp") {
    Write-Host "  GPU Model: $env:GPU_MODEL_PATH"
    Write-Host "  NPU:       $env:NPU_DEVICE"
}
Write-Host "  RAG:       $env:RAG_ENABLED"
Write-Host "  Agents:    agents.yml"

# ── Launch UI ────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Starting Project Overdrive UI..." -ForegroundColor Green
& "$ProjectRoot\venv\Scripts\streamlit.exe" run "$ProjectRoot\app\ui.py"
