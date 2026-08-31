<#
.SYNOPSIS
    Register (or re-register) the quarterly panel-vintage capture with Task
    Scheduler.

.DESCRIPTION
    `scripts\capture_panel_vintage_task.ps1` has named this file since it was
    written and it did not exist, and nothing anywhere in the repository called
    schtasks or Register-ScheduledTask. So the capture was scheduled in
    principle and unscheduled in fact.

    That matters more than a missing convenience. The panel vintage is the only
    instrument in this project that can score the model against REALISED returns
    rather than against the analyst trail it was fitted to, and it needs two
    vintages before it can say anything at all. The first was captured on
    2026-08-31. If the second never fires, every gate in the pipeline goes on
    grading the model against its own input, which is the standing caveat on
    every finding the workflow produces.

    WHAT IT REGISTERS
      * Quarterly, on the 1st of Mar / Jun / Sep / Dec at 06:00 local.
      * Runs the tickler, not the export. The tickler captures if there is a
        fresh export to capture and exits non-zero if there is not -- see the
        run_id guard in capture_panel_vintage.py, which refuses to store one
        export's prices under two dates.
      * NO STORED CREDENTIALS. The task runs as the current user with
        -ExecutionPolicy Bypass, and DB_URL comes from set_env.ps1, which the
        tickler dot-sources. A password in a scheduled task is a password in the
        registry.

    IDEMPOTENT. Re-running replaces the existing registration rather than
    creating a second one; two tasks capturing the same vintage would hit the
    run_id guard, but only after both had run.

.PARAMETER Time
    Local time of day to run. Default 06:00.

.PARAMETER TaskName
    Scheduled-task name. Default 'PML panel vintage capture'.

.PARAMETER Unregister
    Remove the task instead of creating it.

.EXAMPLE
    . .\set_env.ps1
    .\scripts\register_vintage_task.ps1

.EXAMPLE
    .\scripts\register_vintage_task.ps1 -Unregister
#>
[CmdletBinding()]
param(
    [string]$Time = '06:00',
    [string]$TaskName = 'PML panel vintage capture',
    [switch]$Unregister
)

$ErrorActionPreference = 'Stop'

$repo   = Split-Path -Parent $PSScriptRoot
$script = Join-Path $PSScriptRoot 'capture_panel_vintage_task.ps1'

if (-not (Test-Path $script)) {
    throw "missing $script -- this registers that tickler and nothing else"
}

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($Unregister) {
    if ($existing) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "removed scheduled task '$TaskName'"
    } else {
        Write-Host "no scheduled task named '$TaskName'; nothing to remove"
    }
    return
}

# Task Scheduler has no native "quarterly" trigger, so this is a MONTHLY trigger
# restricted to four months. -DaysOfMonth 1 with -Months is the supported form;
# a weekly trigger every 13 weeks would drift off the quarter boundary.
$trigger = New-ScheduledTaskTrigger -Monthly -At $Time -DaysOfMonth 1 `
    -MonthsOfYear March, June, September, December

$action = New-ScheduledTaskAction `
    -Execute 'powershell.exe' `
    -Argument "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$script`"" `
    -WorkingDirectory $repo

# StartWhenAvailable is the one that matters: a laptop asleep on the 1st should
# capture when it wakes, not skip the quarter. There is no catching up later --
# the price and target trails on the MV are unversioned, so a missed vintage is
# a vintage that cannot be reconstructed.
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopIfGoingOnBatteries `
    -AllowStartIfOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1)

if ($existing) {
    Write-Host "replacing the existing '$TaskName' registration"
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask `
    -TaskName $TaskName `
    -Trigger $trigger `
    -Action $action `
    -Settings $settings `
    -Description ("Quarterly panel-vintage capture for the PML Kalman v2 " +
                  "workflow. Runs capture_panel_vintage_task.ps1, which captures " +
                  "only if a fresh export exists and exits non-zero otherwise. " +
                  "Logs to logs\panel_vintage.log.") `
    -RunLevel Limited | Out-Null

$task = Get-ScheduledTask -TaskName $TaskName
$info = Get-ScheduledTaskInfo -TaskName $TaskName
Write-Host ""
Write-Host "registered '$TaskName'"
Write-Host "  action    : $script"
Write-Host "  schedule  : 1 Mar / Jun / Sep / Dec at $Time"
Write-Host "  next run  : $($info.NextRunTime)"
Write-Host "  state     : $($task.State)"
Write-Host "  log       : $(Join-Path $repo 'logs\panel_vintage.log')"
Write-Host ""
Write-Host "Run it once now to prove the wiring:"
Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "It will refuse if the current export was already captured, which is"
Write-Host "the run_id guard working rather than a failure."
