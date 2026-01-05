from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os
import subprocess
import sys

class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        # Compile Metal kernels
        print("Compiling Metal kernels...")
        # Path relative to setup.py
        nx = os.path.join("src", "metalcore", "native")
        src = os.path.join(nx, "core_kernels.metal")
        air = os.path.join(nx, "core_kernels.air")
        lib = os.path.join(nx, "core_kernels.metallib")
        
        # Check if metal compiler is available
        try:
            subprocess.check_call(['xcrun', '-sdk', 'macosx', 'metal', '-c', src, '-o', air])
            subprocess.check_call(['xcrun', '-sdk', 'macosx', 'metallib', air, '-o', lib])
            # Clean up intermediate air file
            if os.path.exists(air):
                os.remove(air)
            print("Successfully compiled core_kernels.metallib")
        except Exception as e:
            print(f"WARNING: Metal compilation failed: {e}")
            print("Will rely on runtime compilation if .metal source is present.")

        super().build_extensions()

setup(
    name='metalcore',
    version='0.1.1',
    author='Kris Bailey',
    author_email='kris@krisbailey.com',
    description='Foundational Metal Linear Algebra Primitives for PyTorch',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    ext_modules=[
        CppExtension(
            name='metalcore_backend',
            sources=['src/metalcore/native/core_mps.mm'],
            extra_compile_args={'cxx': ['-std=c++17', '-fno-objc-arc', '-O3', '-ffast-math']},
            extra_link_args=['-framework', 'Metal', '-framework', 'Foundation'],
        ),
    ],
    cmdclass={
        'build_ext': CustomBuildExtension
    },
    install_requires=['torch>=2.0', 'numpy'],
    include_package_data=True,
    package_data={
        'metalcore': ['native/*.metal', 'native/*.metallib'],
    },
)
