#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda.h>

#include "device_launch_parameters.h"
#include "cuda_runtime.h"

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blockPerGrid = min(32, (N + threadsPerBlock - 1) / threadsPerBlock);

using namespace std;

__host__ void errorHandler(cudaError_t error, const string& msg);

__global__ void kernelScalCalc(const int *v1, const int *v2, int *ans, int n);
__global__ void kernelVectCalc(const int *v1, const int *v2, int *ans, int n);
__global__ void kernelMixedCalc(const int *v1, const int *v2, const int *v3, int *ans, int n);

__host__ void doCudaMalloc(int* cuVec, const size_t size, const string msg);
__host__ void doCudaMemcpyHtD(int* cuVec, vector<int> vec, const size_t size, const string msg);
__host__ void doCudaMemcpyDtH(vector<int> vec, int* cuVec, const size_t size, const string msg);
__host__ void doCudaDeviceSynchronize();
__host__ void doCudaFree(int* cuVec, const string msg);


__host__ void callKernelScal(int* v1, int* v2, int* ans, int len);
__host__ void callKernelVect(int* v1, int* v2, int* ans, int len);
__host__ void callKernelMixed(int* v1, int* v2, int* v3, int* ans, int len);


