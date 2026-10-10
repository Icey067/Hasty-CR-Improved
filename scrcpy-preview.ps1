# Read-only native-resolution preview of the verified Clash Royale instance.
param()
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$python = Join-Path $root '.venvs\buildabot\Scripts\python.exe'
$scrcpy = Join-Path $root 'tools\scrcpy\scrcpy-win64-v4.1\scrcpy.exe'
if (-not (Test-Path -LiteralPath $scrcpy)) {
    throw 'scrcpy v4.1 is missing from tools/scrcpy.'
}
& $python (Join-Path $root 'scripts\emulator.py')
if ($LASTEXITCODE -ne 0) { throw 'Clash Royale device verification failed.' }
$previousAdb = $env:ADB
try {
    $env:ADB = 'C:\Program Files\Netease\MuMuPlayer\nx_device\15.0\shell\adb.exe'
    & $scrcpy '--serial=127.0.0.1:16480' '--no-control' '--no-audio' `
        '--no-downsize-on-error' '--video-codec=h264' '--video-bit-rate=24M' `
        '--print-fps' '--window-title=Clash Royale - scrcpy native' '--window-height=900'
}
finally { $env:ADB = $previousAdb }
