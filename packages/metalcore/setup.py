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
    version='0.1.3',
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
