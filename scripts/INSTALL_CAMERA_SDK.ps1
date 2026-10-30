param([string]$Python = 'python', [switch]$Offline)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$requirements = Join-Path $projectRoot 'requirements-camera.txt'
if ($Offline) {
    $wheel = Join-Path $projectRoot 'third_party/dvsense_sdk/dvsense_driver-1.3.0-cp39-cp39-win_amd64.whl'
    # Project dependencies (including NumPy) must already be installed.
    & $Python -m pip install --no-index --no-deps $wheel
} else {
    & $Python -m pip install -r $requirements
}
if ($LASTEXITCODE -ne 0) { throw "Camera SDK installation failed: $LASTEXITCODE" }
& $Python (Join-Path $projectRoot 'tools/check_camera_sdk.py')
if ($LASTEXITCODE -ne 0) { throw "Camera SDK import check failed: $LASTEXITCODE" }
