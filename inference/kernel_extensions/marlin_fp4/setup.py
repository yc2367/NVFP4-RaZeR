from setuptools import setup
from torch.utils import cpp_extension
import os

os.environ['TORCH_CUDA_ARCH_LIST'] = '12.0a+PTX'


extra_compile_args = {
    "cxx": [
        "-g", 
        "-O3", 
        "-fopenmp", 
        "-lgomp", 
        "-std=c++17",
        # "-DENABLE_BF16"
    ],
    "nvcc": [
        # "-O0", "-G", "-g", # uncomment for debugging
        "-O3",
        "-std=c++17",
        "-DENABLE_BF16",  # TODO
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        # "--threads=8"
        "-g",
        "-lineinfo",
    ],
}


setup(
    name='marlin-fp4',
    version='0.0.1',
    description='Highly optimized FP16xFP4 CUDA matmul kernel.',
    install_requires=['torch'],
    packages=['marlin_fp4'],
    ext_modules=[cpp_extension.CUDAExtension(
        'marlin_fp4_cuda', ['marlin_fp4/marlin_fp4_cuda.cpp', 'marlin_fp4/marlin_fp4_cuda_kernel.cu'],
        extra_compile_args=extra_compile_args
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
