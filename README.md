# Matrix Multiplication

Matrix multiplication implemented in CUDA C++ with both GPU and CPU versions for comparison. For 1000x1000 matrices the GPU computed the solution 6 orders of magnitude faster than the CPU.

Created using Visual Studio 2019, hence the amount of files. Relevant code is in `./CudaMatrixMultiplicationCustom/kernel.cu`

Although it was only tested for square matrices, should work for rectangular ones.

