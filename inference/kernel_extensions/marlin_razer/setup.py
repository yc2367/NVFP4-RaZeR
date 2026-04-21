from setuptools import setup
from torch.utils import cpp_extension
import os

os.environ['TORCH_CUDA_ARCH_LIST'] = '12.0a;12.1a+PTX'


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
    name='marlin-razer',
    version='0.0.1',
    description='Highly optimized FP16xFP4+Negative Zero Remap CUDA matmul kernel.',
    install_requires=['torch'],
    packages=['marlin_razer'],
    ext_modules=[cpp_extension.CUDAExtension(
        'marlin_razer_cuda', ['marlin_razer/marlin_razer_cuda.cpp', 'marlin_razer/marlin_razer_cuda_kernel.cu'],
        extra_compile_args=extra_compile_args
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
