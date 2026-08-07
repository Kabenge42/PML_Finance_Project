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
# Kalman price-target workflow artifact exports. Artifacts land in per-section
# subdirectories under this root (01_data/, 02_eda/, ... 00_misc/); the curated
# bulk frames export to analytics."<stem>" tables + a generated <stem>.sql DDL
# instead of CSV. Anchored at this script's directory so the path is absolute
# regardless of CWD.
$env:KALMAN_PT_RESULTS_DIR = Join-Path $PSScriptRoot "pymc_kalman_filter_pt_results"
# 0 -> emit DDL + CSV without writing to the analytics schema (same fallback
#      happens automatically when the database is unreachable).
$env:KALMAN_PT_SQL_EXPORT = "1"
# 1 -> purge each section subdirectory on first entry so a re-run does not
#      interleave with the previous run's artifacts.
$env:KALMAN_PT_CLEAN_RESULTS = "0"

# Model configuration
$env:MODEL_VERSION = "v9_10"
$env:RANDOM_SEED = "42"

$env:GEIB_DASHBOARD = 'true'

# Performance settings
$env:N_JOBS = "-1"

# PyTensor / PyMC Configuration
# -----------------------------------------------------------------------------
# DEFAULT: pure-Python / numba VM (cxx=""). PyMC 6.x + nutpie use the numba
# backend and do NOT require PyTensor's C linker -- nutpie compiles the logp via
# its own numba path, and pm.model_to_graphviz / shape evaluation run fine on the
# Python VM.
#
# ROOT CAUSE of the historical empty-stderr "Compilation failed (return
# status=1)" (diagnosed 2026-07-10): the g++ *driver* lives in
# C:\msys64\ucrt64\bin, but the actual compiler cc1plus.exe lives in
# ucrt64\lib\gcc\... and loads its DLLs (libgmp, libmpfr, libwinpthread,
# zstd, ...) from ucrt64\bin via PATH. If PATH lacks the ucrt64\bin DIRECTORY
# (e.g. it contains the g++.exe FILE path instead -- a real misconfiguration
# found in the System PATH), cc1plus dies on startup with STATUS_DLL_NOT_FOUND
# (0xC0000135) and zero output: g++ reports status=1 with empty stderr.
# `g++ --version` still works (driver only), which made this look like a
# compiler/ABI problem. The UCRT64 toolchain itself is the correct match for
# MSVC-built CPython (both use the UCRT runtime); verified working on
# Python 3.14.6 + g++ 15.2.0 with a fresh compile cache.
#
# OPT-IN: set  $env:PML_ENABLE_PYTENSOR_C = "1"  BEFORE sourcing this script to
# turn the C backend on (probes UCRT64 then mingw64 for g++.exe, prepends the
# toolchain bin dir to PATH, and sanity-compiles a probe before enabling).
if ($env:PML_ENABLE_PYTENSOR_C -eq "1")
{
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
        # Prepend the toolchain bin DIR (not the g++.exe file!) so cc1plus.exe
        # and linked artifacts resolve libgmp / libmpfr / libwinpthread /
        # libstdc++-6.dll etc. (idempotent -- won't duplicate on repeat sourcing).
        if (-not ($env:Path -split ';' | Where-Object { $_ -ieq $MingwBin }))
        {
            $env:Path = "$MingwBin;$env:Path"
        }

        # Sanity-probe: actually compile something. `g++ --version` succeeding
        # is NOT sufficient -- cc1plus.exe dies silently (0xC0000135, empty
        # stderr) if the toolchain DLLs don't resolve. Fail loudly here rather
        # than mid-sampling with an opaque CompileError.
        $ProbeSrc = Join-Path $env:TEMP "pml_cxx_probe.cpp"
        $ProbeExe = Join-Path $env:TEMP "pml_cxx_probe.exe"
        "int main() { return 0; }" | Set-Content $ProbeSrc
        & $GxxPath $ProbeSrc -o $ProbeExe 2>&1 | Out-Null
        $ProbeOk = ($LASTEXITCODE -eq 0)
        Remove-Item $ProbeSrc, $ProbeExe -Force -ErrorAction SilentlyContinue

        if ($ProbeOk)
        {
            # PYTENSOR_FLAGS parser treats backslashes as escapes and strips them,
            # so C:\msys64\ucrt64\bin\g++.exe mangles into a nonexistent path. Emit
            # the path with forward slashes, which survive the parser and g++/Windows.
            $GxxFwd = $GxxPath -replace '\\', '/'
            $env:PYTENSOR_FLAGS = "floatX=float64,cxx=$GxxFwd"
            Write-Host "PyTensor C backend ENABLED (opt-in) via $GxxFwd (compile probe passed)" -ForegroundColor Yellow
        }
        else
        {
            Write-Warning ("PML_ENABLE_PYTENSOR_C=1 but the g++ compile probe FAILED (exit $LASTEXITCODE). " +
                    "Likely cc1plus.exe cannot load its DLLs -- ensure $MingwBin (the DIRECTORY) is on PATH. " +
                    "Falling back to the pure-Python/numba VM (cxx=).")
            $env:PYTENSOR_FLAGS = "floatX=float64,cxx="
        }
    }
    else
    {
        Write-Warning "PML_ENABLE_PYTENSOR_C=1 but g++.exe not found in C:\msys64\ucrt64\bin or C:\msys64\mingw64\bin -- using pure-Python PyTensor VM."
        $env:PYTENSOR_FLAGS = "floatX=float64,cxx="
    }
}
else
{
    # Robust default: numba VM. nutpie sampling and model_to_graphviz both work.
    $env:PYTENSOR_FLAGS = "floatX=float64,cxx="
    Write-Host "PyTensor backend: pure-Python/numba VM (cxx= empty). Set PML_ENABLE_PYTENSOR_C=1 to opt into the g++ C backend." -ForegroundColor Green
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
Write-Host "KALMAN_PT_RESULTS_DIR: $env:KALMAN_PT_RESULTS_DIR"
Write-Host "KALMAN_PT_SQL_EXPORT: $env:KALMAN_PT_SQL_EXPORT  KALMAN_PT_CLEAN_RESULTS: $env:KALMAN_PT_CLEAN_RESULTS"
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