#include <kernel.cu>
#include <GPULearn.hh>
#include <assert.h>
#include <iostream>

#include <helper_cuda.h>

using namespace std;

GPULearn::GPULearn(float* a_in, int len_a,
		           float* b_in, int len_b) 
{
	// check dim
	assert(len_a == len_b);

	a_h = a_in;
	b_h = b_in;

	length = len_a;

	size_t bytes = length * sizeof(float);


	// allocate device memory
	checkCudaErrors(cudaMalloc(&a_d, bytes));
	checkCudaErrors(cudaMalloc(&b_d, bytes));
	checkCudaErrors(cudaMalloc(&c_d, bytes));

	//cudaError_t err = cudaMalloc((void**) &array_device, size);
	//assert(err == 0);

	checkCudaErrors(cudaMemcpy(a_d, a_h, bytes, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(b_d, b_h, bytes, cudaMemcpyHostToDevice));

	//err = cudaMemcpy(array_device, array_host, size, cudaMemcpyHostToDevice);
	//assert(err == 0);
}

void GPULearn::vectorAdd() {

	dim3 blocks = dim3(256, 1, 1);
	dim3 grids  = dim3(BLK(length, 256), 1, 1);

	kernel_vectorAdd<<< grids, blocks >>>(a_d, b_d, c_d, length);
}

void GPULearn::getData() {
	size_t bytes = length * sizeof(float);
	checkCudaErrors(cudaMemcpy(c_h, c_d, bytes, cudaMemcpyDeviceToHost));
}

void GPULearn::getData_extern(float* c_out, int dim_c) {
	assert(length == dim_c);
	size_t bytes = length * sizeof(float);
	checkCudaErrors(cudaMemcpy(c_out, c_d, bytes, cudaMemcpyDeviceToHost));
}

GPULearn::~GPULearn() {
	checkCudaErrors(cudaFree(a_d));
	checkCudaErrors(cudaFree(b_d));
	checkCudaErrors(cudaFree(c_d));
}
