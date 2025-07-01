# Installing Jecq
This guide provides instructions for installing Jecq on Linux and Windows. You can either install a pre-built package for Python or build the entire C++ library from its source code.

## Option 1: Install from Pre-built Wheels (Recommended)
This is the easiest way to install the Jecq Python package. Pre-built wheels are included with each release.

### Linux

```sh
# This script installs system-level dependencies like BLAS and OpenMP using apt.
./install_requirements.sh

# The wildcard (*) automatically selects the correct version from the directory.
pip install "$(ls ./wheels/linux/jecq-*.whl | head -n1)"
```

### Windows

```sh
# The wildcard (*) automatically selects the correct version from the directory.
pip install (Get-ChildItem .\wheels\windows\jecq-*.whl | Select-Object -First 1 -ExpandProperty FullName)
```

## Option 2: Build from Source
Building from source is recommended for developers who need to modify the C++ core or link against it in their own applications.

For Faiss Developers: If you are already building Faiss, the process for Jecq is nearly identical and supports the same CMake flags.

### Step 1: Install Prerequisites

#### On Linux
The basic requirements are a C++ 17 compiler and a BLAS implementation.

```sh
# This script installs required build dependencies using the 'apt' package manager.
./install_requirements.sh
```

For best performance on Intel CPUs, we also recommend installing the Intel MKL library:

```sh
# This script downloads and installs the Intel MKL library.
./install_mkl.sh
```

#### On Windows
You will need to install several tools to create a build environment. We recommend using a Developer PowerShell for Visual Studio terminal run as Administrator for these steps.

1. Visual Studio Build Tools:
Download and run the Visual Studio Installer. During installation, select the "Desktop Development with C++" workload. This includes the C++ compiler and MSBuild.

2. Intel® oneAPI Math Kernel Library (MKL):
Jecq uses MKL for high-performance computations. Install it from the Intel oneMKL page or use winget.

```sh
winget install --id=Intel.oneMKL -e --accept-package-agreements --accept-source-agreements
```

3. CMake:
This tool generates the build files for Visual Studio. Install it from the CMake download page or use winget.

```sh
winget install -e --id Kitware.CMake
```

4. Python:
Install a recent version of Python from the official Python website or the Microsoft Store.

5. vcpkg (C++ Package Manager):
We use vcpkg to manage C++ dependencies.

```sh
# Clone vcpkg to a permanent location, for example C:\vcpkg
git clone https://github.com/microsoft/vcpkg.git C:\vcpkg

# Run the bootstrap script to build vcpkg
C:\vcpkg\bootstrap-vcpkg.bat

# Integrate vcpkg with MSBuild for all users
vcpkg integrate install
```

6. Set up vcpkg Environment Variable:

Important: The build system finds vcpkg through an environment variable. Setting it in PowerShell with $env:VCPKG_ROOT is temporary and only lasts for the current session. For a permanent, more reliable setup, follow these steps:

* In the Windows Start Menu, type env and select "Edit the system environment variables".
* In the System Properties window, click the "Environment Variables..." button.
* In the "System variables" section, click "New...".
* For "Variable name", enter VCPKG_ROOT.
* For "Variable value", enter the path to your vcpkg installation (e.g., C:\vcpkg).
* Click OK on all windows. You will need to restart your PowerShell terminal for this change to take effect.

7. gflags (Dependency):
Install the gflags library using vcpkg.

```sh
vcpkg install gflags:x64-windows
```

### Step 2: Build the Library
After installing all prerequisites, you can build the project.

#### Quick Build (Using Scripts)
This is the fastest way to compile the library and its Python bindings.

##### On Linux:

```sh
# This script runs cmake to configure and make to compile the project.
./build.sh
```

##### On Windows:

Note: Ensure you are in a Developer PowerShell for Visual Studio where the prerequisites are available.

```sh
# This script runs cmake to configure and msbuild to compile the project.
./build.ps1
```

#### Manual Build (Step-by-Step)
This gives you more control over the build process.

1. Configure with CMake
This step generates the project files for your native build system.

##### On Linux:

```sh
cmake -B build_linux .
```

##### On Windows:

```sh
# This command assumes Visual Studio 2022.
cmake -B build . -G "Visual Studio 17 2022"
```

You can pass Jecq- and Faiss-specific flags here. For example, use `-DJECQ_ENABLE_PYTHON=OFF` to disable the Python bindings and `-DBUILD_TESTING=ON` to enable C++ tests.

2. Compile with Make / MSBuild
This step uses the generated files to compile the C++ code.

##### On Linux:

```sh
# The -j flag enables parallel compilation; adjust the number for your CPU cores.
make -C build_linux -j8
```

##### On Windows:

```sh
# This command builds the entire solution in RelWithDebInfo configuration.
msbuild build/jecq.sln /p:Configuration=RelWithDebInfo
```

3. Build Python Bindings (Optional)
If you enabled Python bindings, this step builds and installs the Python package.

##### On Linux:

```sh
pushd ./build_linux/jecq/python
python3 setup.py install
popd
```

##### On Windows:

```sh
Push-Location ./build/jecq/python
python setup.py install
Pop-Location
```

### Step 3: Test the Build (Optional)
If you configured CMake with `-DBUILD_TESTING=ON`, you can run the C++ test suite to verify the build.

#### On Linux:

```sh
./build_linux/tests/jecq_test
```

#### On Windows:

```sh
# The .exe is located in the RelWithDebInfo folder for a RelWithDebInfo build.
./build/tests/RelWithDebInfo/jecq_test.exe
```

### Step 4: Run the Demos
The demos folder contains sample scripts that compare Jecq and Faiss, allow for hyper-parameterization and more.

#### On Linux:

```sh
# This script prepares the Python environment for the demos.
./build_demo.sh

source ./build_linux/.venv/bin/activate
python3 ./demos/demo_sample_search.py
```

#### On Windows:

```sh
# This script prepares the Python environment for the demos.
./build_demo.ps1

python ./demos/demo_sample_search.py
```