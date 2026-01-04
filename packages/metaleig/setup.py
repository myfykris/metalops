from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

# Helper to find .metal files
def get_metal_files():
    metal_files = []
    for root, dirs, files in os.walk("native"):
        for file in files:
            if file.endswith(".metal"):
                metal_files.append(os.path.join(root, file))
    return metal_files

setup(
    name='metaleig',
    version='0.0.3',
    author='Antigravity',
    description='Metal Eigendecomposition for PyTorch MPS',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    ext_modules=[
        CppExtension(
            name='metaleig_backend',
            sources=['native/eigh_mps.mm'],
            extra_compile_args={'cxx': ['-O3', '-std=c++17']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=['torch'],
    include_package_data=True,
    # Ensure metal files are included in the package
    data_files=[('native', get_metal_files())]
)
