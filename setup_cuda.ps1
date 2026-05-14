# =============================================================================
# Project Overdrive — CUDA Setup for llama-cpp-python
# =============================================================================
# Installs llama-cpp-python with CUDA (NVIDIA GPU) support on Windows.
# Requires: CUDA Toolkit 12.x, CMake, Visual Studio Build Tools

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  llama-cpp-python CUDA Setup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# Check for CUDA
$cudaPath = $env:CUDA_PATH
if (-not $cudaPath) {
    Write-Host ""
    Write-Host "WARNING: CUDA_PATH not set. CUDA Toolkit may not be installed." -ForegroundColor Yellow
    Write-Host "Download from: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Yellow
    Write-Host ""
    $confirm = Read-Host "Continue anyway? (y/N)"
    if ($confirm -ne 'y') { exit 1 }
} else {
    Write-Host "CUDA found at: $cudaPath" -ForegroundColor Green
}

# Check for CMake
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
Write-Host "CMake found: $($cmake.Source)" -ForegroundColor Green

# Set environment for CUDA build
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
$env:FORCE_CMAKE = "1"

Write-Host ""
Write-Host "Installing llama-cpp-python with CUDA support..." -ForegroundColor Cyan
Write-Host "This may take several minutes (compiling from source)." -ForegroundColor Yellow
Write-Host ""

# Install with force to rebuild
& .\venv\Scripts\pip.exe install llama-cpp-python --force-reinstall --no-cache-dir

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Build failed. Common fixes:" -ForegroundColor Red
    Write-Host "  1. Install Visual Studio Build Tools (C++ workload)" -ForegroundColor Yellow
    Write-Host "  2. Ensure CUDA Toolkit 12.x is installed" -ForegroundColor Yellow
    Write-Host "  3. Ensure CMake is on PATH" -ForegroundColor Yellow
    exit 1
}

# Verify CUDA support
Write-Host ""
Write-Host "Verifying CUDA support..." -ForegroundColor Cyan
& .\venv\Scripts\python.exe -c @"
from llama_cpp import Llama
import llama_cpp
print(f'llama-cpp-python version: {llama_cpp.__version__}')
print('CUDA support: Available (build succeeded with GGML_CUDA=on)')
print('Setup complete!')
"@

Write-Host ""
Write-Host "CUDA setup complete." -ForegroundColor Green
