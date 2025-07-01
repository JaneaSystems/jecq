# Copyright (c) 2025 Janea Systems
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

function Test-Exit-Code {
    if (($LASTEXITCODE -ne 0) -and ($LASTEXITCODE -ne $null)) {
        Write-Error "Command failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

# Rebuild if necessary
if (-Not (Test-Path -Path "./build" -PathType Container)) {
    Write-Host "Running build script..."
    & ./build.ps1; Test-Exit-Code
}
else {
    Write-Host "Build directory already exists, skipping build."
}

# Setup environment
./build/.venv/scripts/activate.ps1

if ($env:VIRTUAL_ENV) {
    pip install -r ./demos/requirements.txt; Test-Exit-Code
}
else {
    Write-Error "Virtual environment not activated."
    exit 1
}

# Run demo script
$demoScript = "./demos/demo_sample_search.py"
python $demoScript; Test-Exit-Code