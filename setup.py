from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import torch
import os
import sys

torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib')

# ── Ensure extension is always built in-place for editable installs ──────────
# pip install -e . (PEP 517) does NOT run build_ext --inplace automatically,
# leaving _vvtk_core.so missing.  Force --inplace when 'develop' or 'editable'
# is detected so that a single `pip install -e .` just works.
if 'develop' in sys.argv or 'editable_wheel' in sys.argv:
    if '--inplace' not in sys.argv and 'build_ext' not in sys.argv:
        # inject build_ext --inplace before the original command
        idx = sys.argv.index('develop') if 'develop' in sys.argv else sys.argv.index('editable_wheel')
        sys.argv[idx:idx] = ['build_ext', '--inplace']

setup(
    name='vvtk_dataset',
    version='0.7',
    license='BSD-3-Clause',
    license_files=('LICENSE', 'THIRD_PARTY_NOTICES', 'csrc/LICENSE_zstd', 'csrc/LICENSE_dr_flac'),
    description='High-performance sharded binary dataset library for PyTorch',
    packages=['vvtk_dataset'],
    include_package_data=True,
    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'Operating System :: OS Independent',
    ],
    ext_modules=[
        CppExtension(
            name='_vvtk_core',
            sources=['csrc/vvtk_lib.cpp', 'csrc/zstddeclib.c', 'csrc/dr_flac_impl.c'],
            extra_compile_args=['-O3', '-march=native', '-fopenmp', '-std=c++17'],
            extra_link_args=[
                '-lgomp',
                f'-Wl,-rpath,{torch_lib_dir}',
            ]
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)