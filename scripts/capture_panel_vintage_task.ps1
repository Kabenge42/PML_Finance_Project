<#
.SYNOPSIS
    Quarterly panel-vintage capture, for Windows Task Scheduler.

.DESCRIPTION
    This is a TICKLER, not an unattended pipeline. It captures if there is a
    fresh export to capture, and fails loudly if there is not -- see the run_id
    guard in capture_panel_vintage.py, which refuses to store one export's
    prices under two dates.

    It deliberately does NOT run the export itself. That is an ~11 minute NUTS
    fit which rewrites the analytics tables the GEIB dashboard reads, and
    CLAUDE.md requires the export and the dashboard deploy to ship as a pair.
    Automating that would be automating a decision, so a stale-export run exits
    non-zero and the log says what to do about it.

    Credentials are NOT stored here: DB_URL comes from set_env.ps1, which is
    where it already lives.

    Register with: scripts\register_vintage_task.ps1
    Log:           logs\panel_vintage.log  (UTF-8)
#>
[CmdletBinding()]
param()

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$logDir = Join-Path $repo 'logs'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$log = Join-Path $logDir 'panel_vintage.log'

function Write-Log {
    param([string[]]$Lines)
    # -Encoding utf8 explicitly: the default under Task Scheduler wrote UTF-16,
    # so the log came back as "a s o f _ d a t e" and every em-dash was mojibake.
    $Lines | Add-Content -Path $log -Encoding utf8
}

Write-Log @('', "==== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====")

# Dot-source the project's own environment so DB_URL / PYTENSOR_FLAGS match an
# interactive run exactly. It prompts for nothing.
try {
    . (Join-Path $repo 'set_env.ps1') *> $null
}
catch {
    Write-Log @("could not load set_env.ps1: $_")
    exit 1
}
$env:PYTHONIOENCODING = 'utf-8'

# The PyCharm SDK interpreter is the one with the project's stack installed; the
# repo .venv and the global python are both broken for it.
$py = Join-Path $env:USERPROFILE 'AppData\Local\Python\bin\python.exe'
if (-not (Test-Path $py)) { $py = 'python' }

# 'Continue', not 'Stop': the capture signals a refused vintage by EXITING
# NON-ZERO with an explanatory message on stderr. Under 'Stop' that stderr became
# a terminating PowerShell error, so a well-behaved refusal was logged as
# "wrapper error" and the real message was buried.
$ErrorActionPreference = 'Continue'
# PowerShell decodes a native command's output with the CONSOLE codepage, not
# PYTHONIOENCODING, so the script's em-dashes arrived as mojibake in the log.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$output = & $py (Join-Path $repo 'scripts\capture_panel_vintage.py') 2>&1
$rc = $LASTEXITCODE
Write-Log ($output | ForEach-Object { $_.ToString() })

if ($rc -ne 0) {
    Write-Log @(
        "capture exited $rc."
        "If the message above names a run_id already captured, the export is"
        "unchanged and the guard refused rather than storing duplicate prices."
        "To capture a real one:  python pymc_kalman_filter_pt_v2.py --write"
        "then re-run this task."
    )
}
exit $rc
