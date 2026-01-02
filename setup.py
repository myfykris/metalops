from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

setup(
    name='metalsvd',
    version='0.0.1',
    author='Kris Bailey via Antigravity',
    author_email='antigravity@gemini.com',
    description='Batched One-Sided Jacobi SVD on Metal',
    long_description='A PyTorch extension implementing One-Sided Jacobi SVD on macOS Metal.',
    ext_modules=[
        CppExtension(
            name='metalsvd_backend',
            sources=['native/svd_mps.mm'],
            extra_compile_args={'cxx': ['-std=c++17', '-fno-objc-arc']},
            extra_link_args=['-framework', 'Metal', '-framework', 'Foundation'],
            # PyTorch's CppExtension on Mac usually handles -framework Metal if we include ObjC++.
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    packages=['metalsvd'],
    package_dir={'': 'src'},
    package_data={
        'metalsvd': ['src/*.metal'],  # We might want to install the metal shader if we load it from file
    },
    # Making sure src/ is included if we want to package it, though usually we just compile .mm
)
