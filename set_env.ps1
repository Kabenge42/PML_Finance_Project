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
# Disable C backend to avoid MinGW/MSVC ABI mismatch and libgcc 15 linking issues.
# PyTensor will use its pure-Python VM (functionally identical, ~2-3x slower for large MCMC).
# Setting cxx= (empty) disables C compilation entirely.
$env:PYTENSOR_FLAGS = "device=cpu,floatX=float64,cxx="

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


