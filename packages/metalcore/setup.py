from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

def get_metal_files():
    metal_files = []
    for root, dirs, files in os.walk("native"):
        for file in files:
            if file.endswith(".metal"):
                metal_files.append(os.path.join(root, file))
    return metal_files

setup(
    name='metalcore',
    version='0.0.1',
    author='Kris Bailey via Antigravity',
    description='Foundational Metal Linear Algebra Primitives for PyTorch',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    ext_modules=[
        CppExtension(
            name='metalcore_backend',
            sources=['native/core_mps.mm'],
            extra_compile_args={'cxx': ['-std=c++17', '-fno-objc-arc', '-O3', '-ffast-math']},
            extra_link_args=['-framework', 'Metal', '-framework', 'Foundation'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=['torch>=2.0', 'numpy'],
    include_package_data=True,
    data_files=[('native', get_metal_files())]
)
