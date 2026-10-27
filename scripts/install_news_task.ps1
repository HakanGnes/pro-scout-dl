# Registers a Windows scheduled task that runs `python -m etl collect-news` every hour.
#   powershell -ExecutionPolicy Bypass -File scripts\install_news_task.ps1
# Remove it again with:
#   Unregister-ScheduledTask -TaskName ProScout-CollectNews -Confirm:$false
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\pythonw.exe"   # pythonw: no console window every hour
$log = Join-Path $repo "data\raw\news\collect.log"
if (-not (Test-Path $python)) { throw "Not found: $python (create the virtual environment first)" }
if (-not (Test-Path (Join-Path $repo "etl\news\feeds.json"))) { throw "Run first: python -m etl probe-news" }

$action = New-ScheduledTaskAction -Execute $python -Argument "-m etl collect-news --log-file `"$log`"" -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Hours 1) `
    -RepetitionDuration (New-TimeSpan -Days 3650)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 15)
Register-ScheduledTask -TaskName "ProScout-CollectNews" -Action $action -Trigger $trigger -Settings $settings `
    -Description "Pro-Scout DL: archive Turkish football news feeds every hour" -Force | Out-Null
Write-Host "Registered ProScout-CollectNews (every hour, first run in a minute). Log: $log"
