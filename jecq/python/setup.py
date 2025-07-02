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

from __future__ import print_function

import os
import platform
import shutil

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext

# make the jecq python package dir
shutil.rmtree("jecq", ignore_errors=True)
os.mkdir("jecq")
shutil.copyfile("__init__.py", "jecq/__init__.py")
shutil.copyfile("loader.py", "jecq/loader.py")
shutil.copyfile("class_wrappers.py", "jecq/class_wrappers.py")

is_windows = platform.system() == "Windows"

ext = ".pyd" if is_windows else ".so"
build_type = os.environ.get("JECQ_BUILD_TYPE", "RelWithDebInfo")
prefix = f"{build_type}/" * is_windows

swigjecq_generic_lib = f"{prefix}_swigjecq{ext}"
swigjecq_avx2_lib = f"{prefix}_swigjecq_avx2{ext}"
swigjecq_avx512_lib = f"{prefix}_swigjecq_avx512{ext}"
swigjecq_avx512_spr_lib = f"{prefix}_swigjecq_avx512_spr{ext}"
swigjecq_sve_lib = f"{prefix}_swigjecq_sve{ext}"

found_swigjecq_generic = os.path.exists(swigjecq_generic_lib)
found_swigjecq_avx2 = os.path.exists(swigjecq_avx2_lib)
found_swigjecq_avx512 = os.path.exists(swigjecq_avx512_lib)
found_swigjecq_avx512_spr = os.path.exists(swigjecq_avx512_spr_lib)
found_swigjecq_sve = os.path.exists(swigjecq_sve_lib)

assert (
    found_swigjecq_generic
    or found_swigjecq_avx2
    or found_swigjecq_avx512
    or found_swigjecq_avx512_spr
    or found_swigjecq_sve
), (
    f"Could not find {swigjecq_generic_lib} or "
    f"{swigjecq_avx2_lib} or {swigjecq_avx512_lib} or "
    f"{swigjecq_avx512_spr_lib} or {swigjecq_sve_lib}. "
    f"Jecq may not be compiled yet."
)

libs = []

if found_swigjecq_generic:
    print(f"Copying {swigjecq_generic_lib}")
    shutil.copyfile("swigjecq.py", "jecq/swigjecq.py")
    shutil.copyfile(swigjecq_generic_lib, f"jecq/_swigjecq{ext}")
    libs.append("_swigjecq")

if found_swigjecq_avx2:
    print(f"Copying {swigjecq_avx2_lib}")
    shutil.copyfile("swigjecq_avx2.py", "jecq/swigjecq_avx2.py")
    shutil.copyfile(swigjecq_avx2_lib, f"jecq/_swigjecq_avx2{ext}")
    libs.append("_swigjecq_avx2")

if found_swigjecq_avx512:
    print(f"Copying {swigjecq_avx512_lib}")
    shutil.copyfile("swigjecq_avx512.py", "jecq/swigjecq_avx512.py")
    shutil.copyfile(swigjecq_avx512_lib, f"jecq/_swigjecq_avx512{ext}")
    libs.append("_swigjecq_avx512")


if found_swigjecq_avx512_spr:
    print(f"Copying {swigjecq_avx512_spr_lib}")
    shutil.copyfile("swigjecq_avx512_spr.py", "jecq/swigjecq_avx512_spr.py")
    shutil.copyfile(swigjecq_avx512_spr_lib, f"jecq/_swigjecq_avx512_spr{ext}")
    libs.append("_swigjecq_avx512_spr")


if found_swigjecq_sve:
    print(f"Copying {swigjecq_sve_lib}")
    shutil.copyfile("swigjecq_sve.py", "jecq/swigjecq_sve.py")
    shutil.copyfile(swigjecq_sve_lib, f"jecq/_swigjecq_sve{ext}")
    libs.append("_swigjecq_sve")

ext_modules = [Extension(f"jecq.{mod}", sources=[]) for mod in libs]


# 2) CopyBuildExt just copies the .so into the build directory without compiling
class CopyBuildExt(_build_ext):
    def run(self):
        # skip the normal compiler run
        for extension in self.extensions:
            self.build_extension(extension)

    def build_extension(self, extension):
        name = extension.name.split(".", 1)[1]  # e.g. "_swigjecq"
        src = f"{prefix}{name}{ext}"
        if not os.path.exists(src):
            raise FileNotFoundError(f"Binary output '{src}' not found")
        dst = self.get_ext_fullpath(extension.name)
        self.mkpath(os.path.dirname(dst))
        shutil.copyfile(src, dst)
        print(f"Copied {src} to {dst}")


long_description = """
Jecq is a library for efficient similarity search based on the Faiss library.
"""

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    """Custom bdist_wheel to ensure that the root is not pure Python."""

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False

except ImportError:
    print("Not using custom bdist_wheel; wheel package not installed")

setup(
    name="jecq",
    version="1.0.0",
    description="A Faiss-based library for efficient similarity search "
    "and clustering of dense vectors",
    long_description=long_description,
    url="https://github.com/JaneaSystems/jecq",
    license="MIT",
    keywords="search nearest neighbors",
    install_requires=["numpy", "packaging", "faiss-cpu"],
    packages=["jecq"],
    package_data={
        "jecq": ["*.pyd"] if is_windows else [],
    },
    zip_safe=False,
    ext_modules=[] if is_windows else ext_modules,
    cmdclass={"bdist_wheel": bdist_wheel, "build_ext": CopyBuildExt},
)
