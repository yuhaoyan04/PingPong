param([string]$Python = 'python')

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$requirements = Join-Path $projectRoot 'requirements-camera.txt'
& $Python -m pip install -r $requirements
if ($LASTEXITCODE -ne 0) { throw "Camera SDK installation failed: $LASTEXITCODE" }
& $Python (Join-Path $projectRoot 'tools/check_camera_sdk.py')
if ($LASTEXITCODE -ne 0) { throw "Camera SDK import check failed: $LASTEXITCODE" }
