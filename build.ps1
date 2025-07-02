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

Write-Output "Building and installing JECQ from source..."

function Test-Exit-Code {
  if (($LASTEXITCODE -ne 0) -and ($LASTEXITCODE -ne $null)) {
    Write-Error "Command failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
  }
}

function Add-PathEntry {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Folder
  )

  $currentEntries = $env:Path -split ';'

  if (-not ($currentEntries -contains $Folder)) {
    $env:Path += ";$Folder"
  }
}

$buildType = 'RelWithDebInfo'
Write-Host "Building ..."

# Deactivate any existing virtual environment
if ($env:VIRTUAL_ENV) {
  "$env:VIRTUAL_ENV\Scripts\deactivate"
}

# Clean previous build
if (Test-Path build) {
  Remove-Item -Recurse -Force build
}

# Create build directory
mkdir build
Push-Location build

# Setup virtual environment
Write-Output "Setting up virtual environment..."
Push-Location .
python -m venv .venv; Test-Exit-Code
./.venv/scripts/activate.ps1; Test-Exit-Code
python -m pip install --upgrade pip; Test-Exit-Code
pip install -r ../jecq/python/requirements.txt; Test-Exit-Code
Pop-Location

# Save the original VCPKG_ROOT if defined
$originalVcpkgRoot = $env:VCPKG_ROOT -replace '\\','/'

# Set up Intel oneAPI environment
if (-not ($env:SETVARS_COMPLETED -eq '1')) {
  Write-Host "Setting up Intel oneAPI environment..."
  $pgfx86 = ${env:ProgramFiles(x86)}
  $oneAPIDir = "$pgfx86\Intel\oneAPI"
  function Invoke-EnvBatch {
    param([Parameter(Mandatory = $true)] [string] $BatchFile)

    # run the batch, then run `set` and pipe each VAR=VALUE back into PS
    cmd /c "`"$BatchFile`" && set" |
    ForEach-Object {
      if ($_ -match '^(?<n>[^=]+)=(?<v>.*)$') {
        [System.Environment]::SetEnvironmentVariable(
          $matches['n'], $matches['v'], 'Process'
        )
      }
    }
  }

  Invoke-EnvBatch "$oneAPIDir\setvars.bat"
}
else {
  Write-Host "Intel oneAPI environment already set up."
}

# At the end, restore the original VCPKG_ROOT if it was defined
if ($null -ne $originalVcpkgRoot) {
  $env:VCPKG_ROOT = $originalVcpkgRoot
  Write-Host "Restored VCPKG_ROOT to $originalVcpkgRoot"
  # Run again integrate install
  & "$env:VCPKG_ROOT/bootstrap-vcpkg.bat"
  & "$env:VCPKG_ROOT/vcpkg" integrate install
}

# Configure with CMake
Write-Output "Running CMake..."
$mklLib = "$env:MKLROOT\lib"

cmake ../ -G "Visual Studio 17 2022" `
  "-DCMAKE_BUILD_TYPE=$buildType" "-DBUILD_TESTING=ON" `
  "-DFAISS_USE_LTO=ON" "-DFAISS_OPT_LEVEL=avx2" `
  "-DJECQ_ENABLE_PYTHON=ON" "-DFAISS_ENABLE_GPU=OFF" `
  "-DBLA_VENDOR=Intel10_64_dyn" `
  "-DMKL_LIBRARIES=$mklLib\mkl_intel_lp64.lib;$mklLib\mkl_sequential.lib;$mklLib\mkl_core.lib" `
  "-DVCPKG_TARGET_TRIPLET=x64-windows" `
  "-DCMAKE_TOOLCHAIN_FILE=`"$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake`""
Test-Exit-Code

# Build tests project
Write-Output "Building jecq_test..."
msbuild tests\jecq_test.vcxproj `
  /p:Configuration=$buildType `
  /m

# Build Python SWIG wrapper
Write-Output "Building SWIG wrapper..."
msbuild jecq\python\swigjecq.vcxproj `
  /p:Configuration=$buildType `
  /m
Test-Exit-Code
msbuild jecq\python\swigjecq_avx2.vcxproj `
  /p:Configuration=$buildType `
  /m
Test-Exit-Code

# Update paths so dynamic libraries can be found
Add-PathEntry "$env:MKLROOT\bin"
Add-PathEntry "$env:CMPLR_ROOT\bin"

# Run tests executable
$testExe = "./tests/$buildType/jecq_test.exe"
Write-Host "Running tests: $testExe"
& "$testExe"
Test-Exit-Code

# Install Python package
Write-Output "Installing Python package..."
Push-Location jecq/python
pip install wheel; Test-Exit-Code
python setup.py bdist_wheel; Test-Exit-Code
Get-ChildItem dist/jecq*.whl | ForEach-Object { pip install $_.FullName }
Pop-Location

# Verify installation
python -c "import jecq;jecq.IndexJecq();jecq.IndexIVFJecq()"; Test-Exit-Code
python -c "import jecq;assert jecq.IndexJecq.__module__ == 'jecq.swigjecq_avx2'"; Test-Exit-Code

# Update wheels
Write-Output "Updating wheels..."
# Create wheels directory if it doesn't exist
if (-not (Test-Path ../wheels/windows)) {
  New-Item -ItemType Directory -Path ../wheels/windows
}
Get-ChildItem -R jecq*.whl | Copy-Item -Destination ../wheels/windows/

# Open solution for further work
if (-not $env:JECQ_CI) {
  Start-Process devenv.exe "./jecq.sln"
}
Pop-Location
