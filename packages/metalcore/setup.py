from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os
import subprocess
import sys
import re

# Read version from __init__.py (single source of truth)
def get_version():
    init_path = os.path.join("src", "metalcore", "__init__.py")
    with open(init_path, "r") as f:
        content = f.read()
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if match:
        return match.group(1)
    raise RuntimeError("Could not find __version__ in __init__.py")

VERSION = get_version()

class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        # Compile Metal kernels
        print("Compiling Metal kernels...")
        # Path relative to setup.py
        nx = os.path.join("src", "metalcore", "native")
        lib = os.path.join(nx, "core_kernels.metallib")
        
        # Find all .metal files
        metal_files = [f for f in os.listdir(nx) if f.endswith('.metal')]
        air_files = []
        
        try:
            # Compile each .metal to .air
            for metal_file in metal_files:
                src_path = os.path.join(nx, metal_file)
                air_path = os.path.join(nx, metal_file.replace('.metal', '.air'))
                subprocess.check_call(['xcrun', '-sdk', 'macosx', 'metal', '-c', src_path, '-o', air_path])
                air_files.append(air_path)
            
            # Link all .air to single .metallib
            subprocess.check_call(['xcrun', '-sdk', 'macosx', 'metallib'] + air_files + ['-o', lib])
            
            # Clean up intermediate air files
            for air in air_files:
                if os.path.exists(air):
                    os.remove(air)
            print(f"Successfully compiled {len(metal_files)} metal files to core_kernels.metallib")
        except Exception as e:
            print(f"WARNING: Metal compilation failed: {e}")
            print("Will rely on runtime compilation if .metal source is present.")

        super().build_extensions()

setup(
    name='metalcore',
    version=VERSION,
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
