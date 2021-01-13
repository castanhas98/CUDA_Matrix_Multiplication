#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <chrono>

// Debugging, printf works instide kernel, whereas std::cout doesn't, apparently
#include <stdio.h>

#define N 1000
#define NN N * N
// I is tricky because due to the way the code is implemented, it's not the same matrix being multiplied I times, we're just adding to what was already in C
#define I 10
#define BLOCK_SIZE N

// Assert-style error-handling function, taken from https://stackoverflow.com/a/14038590
#define gpuErrorCheck(expr) { gpuAssert((expr), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true)
{
	if (code != cudaSuccess)
	{
		std::cerr << "GPUAssert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
		if (abort) exit(code);
	}
}


// matrix operations assuming matrices with dimensions n x m -> n rows, m columns


void fillMatrix(double* matrix, int n, int m, double coef)
{
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < m; ++j)
			matrix[i * m + j] = coef * (i * m + j);
}

void fillMatrixZeros(double* matrix, int n, int m)
{
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < m; ++j)
			matrix[i * m + j] = 0.0;
}

void printMatrix(double* matrix, int n, int m)
{
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < m; ++j)
			std::cout << matrix[i * m + j] << (j == m - 1 ? "\n" : ", ");

	std::cout << "\n";
}



// performs simple matrix multiplication C = A * B, with A (n x m), B (m x l) and therefore C (n x l)
// assumes C is initialised with zeros
void matrixMultiplicationCPU(const double* A, const double* B, double* C, const int n, const int m, const int l)
{
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < l; ++j)
			for (int k = 0; k < m; ++k)
				C[i * l + j] += A[i * m + k] * B[k * l + j];
}

// matrix multiplication on gpu
// blockIdx.x is equivalent to i in the cpu example
// threadIdx.x is equivalent to j in the cpu example
// blockDim.x is equivalent to m in the cpu example
// assumes that C is zero-filled
__global__ void matrixMultiplicationGPU(const double* A, const double* B, double* C, const int n, const int m, const int l)
{
	// to try: add whole B to memory shared in each block, since the whole B is used to compute each row of C
	int C_index = threadIdx.x + blockIdx.x * blockDim.x;

	// stores one row from matrix A that is used to compute all the entries in the same row from matrix C
	__shared__ double A_row[BLOCK_SIZE];

	// adding whole row from A matrix to shared memory
	for (int k = 0; k < m; ++k)
		A_row[k] = A[blockIdx.x * m + k];

	// synchronising threads
	__syncthreads();

	for (int k = 0; k < m; ++k)
		C[C_index] += A_row[k] * B[k * l + threadIdx.x];

}

// didn't yield much difference with 1000x1000 matrices
__global__ void matrixMultiplicationGPUNoSharedMemory(const double* A, const double* B, double* C, const int n, const int m, const int l)
{
	// to try: add whole B to memory shared in each block, since the whole B is used to compute each row of C
	int C_index = threadIdx.x + blockIdx.x * blockDim.x;

	for (int k = 0; k < m; ++k)
		C[C_index] += A[blockIdx.x * m + k] * B[k * l + threadIdx.x];

}

// main
int main()
{
	double* a = new double[NN];
	double* b = new double[NN];
	double* c = new double[NN];

	fillMatrix(a, N, N, 1.0);
	fillMatrix(b, N, N, 2.0);
	fillMatrixZeros(c, N, N);

	// printMatrix(a, N, N);
	// printMatrix(b, N, N);
	// printMatrix(c, N, N);

	std::cout << "Multiplying Square Matrices with dimensions " << N << " x " << N << " on the CPU " << I << " times" << std::endl;
	auto tic(std::chrono::system_clock::now());
	for(int i = 0; i < I; ++i)
		matrixMultiplicationCPU(a, b, c, N, N, N);
	auto toc(std::chrono::system_clock::now());
	std::cout << std::chrono::duration<double>(toc - tic).count() << " seconds" << std::endl;

	// printMatrix(c, N, N);


	// preparing matrix multiplication on the GPU
	double* d_a, * d_b, * d_c;
	double* c_dest = new double[NN];
	int size = NN * sizeof(double);
	gpuErrorCheck( cudaMalloc((void**) &d_a, size) );
	gpuErrorCheck( cudaMalloc((void**) &d_b, size) );
	gpuErrorCheck( cudaMalloc((void**) &d_c, size) );

	gpuErrorCheck( cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice) );
	gpuErrorCheck( cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice) );

	std::cout << "Multiplying Square Matrices with dimensions " << N << " x " << N << " on the GPU" << I << " times" << std::endl;
	tic = std::chrono::system_clock::now();
	for (int i = 0; i < I; ++i)
		// number of blocks -> (number of entries) / number of (threads/block)
		// number of threads/block -> number of columns
		matrixMultiplicationGPU << < NN / N, N >> > (d_a, d_b, d_c, N, N, N);
		// testing without shared memory
		// matrixMultiplicationGPUNoSharedMemory << < NN / N, N >> > (d_a, d_b, d_c, N, N, N);
	toc = std::chrono::system_clock::now();
	std::cout << std::chrono::duration<double>(toc - tic).count() << " seconds" << std::endl;
	gpuErrorCheck( cudaPeekAtLastError() );
	gpuErrorCheck( cudaMemcpy(c_dest, d_c, size, cudaMemcpyDeviceToHost) );

	// printMatrix(c_dest, N, N);

	delete[] a;
	delete[] b;
	delete[] c;
	delete[] c_dest;

	gpuErrorCheck( cudaFree(d_a) );
	gpuErrorCheck( cudaFree(d_b) );
	gpuErrorCheck( cudaFree(d_c) );

	
	return 0;
}