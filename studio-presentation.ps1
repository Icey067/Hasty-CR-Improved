# Fast emulator-window gameplay plus a narrow decision rail, recorded as a
# 1360x1920 60fps presentation clip.  Do not use the device screenrecord
# stream here: it shares ADB with the bot and can collapse to single-digit fps.
#
#   .\studio-presentation.ps1
#   .\studio-presentation.ps1 -MatchCount 1 -Brain rl

param(
    [Alias('Matches')][int]$MatchCount = 5,
    [ValidateSet('rules', 'rl')][string]$Brain = 'rules',
    [string]$Checkpoint = 'checkpoints\sprint4_baseline\pilot_best_4392960.pt'
)

$root = $PSScriptRoot
$python = Join-Path $root '.venvs\buildabot\Scripts\python.exe'
$botArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
             (Join-Path $root 'run.ps1'), '-MatchCount', $MatchCount,
             '-Brain', $Brain, '-Checkpoint', $Checkpoint, '-FullRes')
Start-Process -FilePath 'powershell.exe' -ArgumentList $botArgs -WorkingDirectory $root
Start-Sleep -Seconds 2
Push-Location $root
try {
    & $python -u -m scripts.studio --layout recording --record --fps 60 --record-fps 60 --window-capture --no-adb-fallback --detect --detect-fps 12 --crf 16 --preset veryfast
}
finally {
    Pop-Location
}
