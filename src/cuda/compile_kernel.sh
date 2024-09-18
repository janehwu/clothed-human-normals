nvcc -Xcompiler -fPIC -shared -o raster_backprop_kernel.so raster_cuda_kernels.cu
nvcc -Xcompiler -fPIC -shared -o mt_backprop_kernel.so mt_cuda_kernels.cu
