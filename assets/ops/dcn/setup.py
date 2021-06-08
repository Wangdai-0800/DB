from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

nvcc_args = ['--gpu-architecture=compute_70','--gpu-code=sm_70']
setup(
    name='deform_conv',
    ext_modules=[
        CUDAExtension('deform_conv_cuda', [
            'src/deform_conv_cuda.cpp',
            'src/deform_conv_cuda_kernel.cu',
        ], extra_compile_args={'cxx': ['-O2',],'nvcc':['--gpu-architecture=compute_70','--gpu-code=sm_70','-O3','-I./cutlass/','-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__']}),
        CUDAExtension('deform_pool_cuda', [
            'src/deform_pool_cuda.cpp', 'src/deform_pool_cuda_kernel.cu'
        ], extra_compile_args={'cxx': ['-O2',],'nvcc':['--gpu-architecture=compute_70','--gpu-code=sm_70','-O3','-I./cutlass/','-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__']}),
    ],
    cmdclass={'build_ext': BuildExtension})
