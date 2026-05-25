# =============================================================================
# PML Finance Project - Environment Variables Setup
# =============================================================================
# PowerShell script to set environment variables for the current session
#
# Usage:
#   .\set_env.ps1
#   or
#   . .\set_env.ps1  (dot-source to persist in current session)
#
# To make permanent (user-level):
#   [Environment]::SetEnvironmentVariable("VAR_NAME", "value", "User")
# =============================================================================

# Disable colored output
$env:NO_COLOR = "1"

# Directory paths
$env:DATA_DIR = "data"
$env:MODEL_DIR = "regression"
$env:CACHE_DIR = ".cache"
$env:OUTPUT_DIR = "outputs"

# Model configuration
$env:MODEL_VERSION = "v9_10"
$env:RANDOM_SEED = "42"

$env:GEIB_DASHBOARD = 'true'

# Performance settings
$env:N_JOBS = "-1"

# PyTensor / PyMC Configuration
# -----------------------------------------------------------------------------
# Enable the C backend via the MSYS2 MinGW g++ toolchain. This is required for
# the nutpie Rust NUTS sampler and for any C-backed PyTensor graph compilation.
# Probe UCRT64 first (matches the `MinGW` system variable and the MSVC-built
# UCRT C runtime used by modern Python wheels), then fall back to mingw64.
# If neither is found, fall back to the pure-Python VM (cxx="") so the rest
# of the pipeline still works.
$CandidateBinDirs = @(
    "C:\msys64\ucrt64\bin",
    "C:\msys64\mingw64\bin"
)
$GxxPath = $null
$MingwBin = $null
foreach ($Dir in $CandidateBinDirs)
{
    $Candidate = Join-Path $Dir "g++.exe"
    if (Test-Path $Candidate)
    {
        $GxxPath = $Candidate
        $MingwBin = $Dir
        break
    }
}

if ($GxxPath)
{
    # Prepend the toolchain bin dir so libstdc++-6.dll / libgcc_s_seh-1.dll
    # resolve at link time (idempotent — won't duplicate on repeat sourcing).
    if (-not ($env:Path -split ';' | Where-Object { $_ -ieq $MingwBin }))
    {
        $env:Path = "$MingwBin;$env:Path"
    }
    $env:PYTENSOR_FLAGS = "device=cpu,floatX=float64,cxx=$GxxPath"
    Write-Host "PyTensor C backend ENABLED via $GxxPath" -ForegroundColor Green
}
else
{
    Write-Warning "g++.exe not found in C:\msys64\ucrt64\bin or C:\msys64\mingw64\bin -- falling back to pure-Python PyTensor VM (nutpie will be unavailable)."
    $env:PYTENSOR_FLAGS = "device=cpu,floatX=float64,cxx="
}

# Logging configuration
$env:LOG_LEVEL = "INFO"
$env:TF_CPP_MIN_LOG_LEVEL = "2"

# Database connection (update with your actual credentials)
# TODO: Update credentials before use or set via secure credential management
$env:DB_URL = "postgresql+psycopg2://postgres:bItcfiTg142!@localhost:5432/postgres"

Write-Host "Environment variables set successfully!" -ForegroundColor Green
Write-Host "NO_COLOR: $env:NO_COLOR"
Write-Host "DATA_DIR: $env:DATA_DIR"
Write-Host "OUTPUT_DIR: $env:OUTPUT_DIR"
Write-Host "MODEL_DIR: $env:MODEL_DIR"
Write-Host "RANDOM_SEED: $env:RANDOM_SEED"
Write-Host "DB_URL: $env:DB_URL"
Write-Host "GEIB_DASHBOARD: $env:GEIB_DASHBOARD"
Write-Host "N_JOBS: $env:N_JOBS"
Write-Host "LOG_LEVEL: $env:LOG_LEVEL"
Write-Host "TF_CPP_MIN_LOG_LEVEL: $env:TF_CPP_MIN_LOG_LEVEL"
Write-Host "MODEL_VERSION: $env:MODEL_VERSION"
Write-Host "CACHE_DIR: $env:CACHE_DIR"
Write-Host "PYTENSOR_FLAGS: $env:PYTENSOR_FLAGS"