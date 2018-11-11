#include "kernel.cuh"

void errorHandler(cudaError_t error, const string& msg) {
	if (error != cudaSuccess) {
		cerr << msg << endl;
		exit(0);
	}
}

__global__ void kernelScalCalc(const int *v1, const int *v2, int *ans, int n) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < n) {
		ans[tid] = v1[tid] * v2[tid];
	}
}

__global__ void kernelVectCalc(const int *v1, const int *v2, int *ans, int n) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < n) {
		ans[tid] = v1[(tid + 1) % 3] * v2[(tid + 2) % 3] - v1[(tid + 2) % 3] * v2[(tid + 1) % 3];
	}
}

__global__ void kernelMixedCalc(const int *v1, const int *v2, const int *v3, int *ans, int n) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < n) {
		ans[tid] = v3[tid] * (v1[(tid + 1) % 3] * v2[(tid + 2) % 3] - v1[(tid + 2) % 3] * v2[(tid + 1) % 3]);
	}
}

__host__ void doCudaMalloc(int* cuVec, const size_t size, const string msg) {
	errorHandler(cudaMalloc(&cuVec, size), msg);
}
__host__ void doCudaMemcpyHtD(int* cuVec, vector<int> vec, const size_t size, const string msg) {
	errorHandler(cudaMemcpy(cuVec, vec.data(), size, cudaMemcpyHostToDevice), msg);
}
__host__ void doCudaMemcpyDtH(vector<int> vec, int* cuVec, const size_t size, const string msg) {
	errorHandler(cudaMemcpy(vec.data(), cuVec, size, cudaMemcpyDeviceToHost), msg);
}
__host__ void doCudaDeviceSynchronize() {
	cudaDeviceSynchronize();
}
__host__ void doCudaFree(int* cuVec, const string msg) {
	errorHandler(cudaFree(cuVec), msg);
}


__host__ void callKernelScal(int* v1, int* v2, int* ans, int len) {
	kernelScalCalc <<< blockPerGrid, threadsPerBlock >>> (v1, v2, ans, len);
}
__host__ void callKernelVect(int* v1, int* v2, int* ans, int len) {
	kernelVectCalc << < blockPerGrid, threadsPerBlock >> > (v1, v2, ans, len);
}
__host__ void callKernelMixed(int* v1, int* v2, int* v3, int* ans, int len) {
	kernelMixedCalc << < blockPerGrid, threadsPerBlock >> > (v1, v2, v3, ans, len);
}


