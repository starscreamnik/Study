#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <cuda.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blockPerGrid = min(32, (N + threadsPerBlock - 1) / threadsPerBlock);

void errorHandler(cudaError_t error, const string& msg) {
	if (error != cudaSuccess) {
		cerr << msg << endl;
		exit(0);
	}
}

void input(ifstream& in, const int dim, int*& m1, int*& m2, int*& mAns, int*& v, int*& vAns) {
	for (int i = 0; i < dim*dim; i++) in >> m1[i];
	for (int i = 0; i < dim*dim; i++) in >> m2[i];
	for (int i = 0; i < dim; i++) in >> v[i];
	memset(vAns, 0, dim * sizeof(int));
	memset(mAns, 0, dim*dim * sizeof(int));
}

void outputMxM(const int dim, int *a, int *b, int *ans) {
	cout << "============Matrix on Matrix=============" << endl;
	cout << "Parallel:\n";
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			cout << ans[i*dim + j] << " ";
		}
		cout << endl;
	}

	cout << "Consistent:\n";
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			int c = 0;
			for (int k = 0; k < dim; k++) {
				c += a[dim*i + k] * b[dim*k + j];
			}
			cout << c << " ";
		}
		cout << "\n";
	}
	cout << endl;
}

void outputMxV(const int dim, int *m, int *v, int *ans) {
	cout << "============Matrix on Vector=============" << endl;
	cout << "Parallel:\n";
	for (int i = 0; i < dim; i++) {
		cout << ans[i] << " ";
	}
	cout << endl;

	cout << "Consistent:\n";
	for (int i = 0; i < dim; i++) {
		int c = 0;
		for (int j = 0; j < dim; j++) {
			c += m[i*dim + j] * v[j];
		}
		cout << c << " ";
	}
	cout << endl;
}

__device__ int findRow(int tid, const int n) {
	int counter = 0;
	while (tid >= n) {
		tid -= n;
		counter++;
	}
	return counter;
}

__global__ void MxMKernel(const int *m1, const int *m2, int *c, int n) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	int i = tid, j = tid;
	if (tid < n*n) {
		if (tid >= n) {
			j %= n;
			i = findRow(tid, n);
		}
		else i = 0;

		for (int k = 0; k < n; k++)
			c[tid] += m1[i*n + k] * m2[k*n + j];
	}
}

__global__ void MxVKernel(const int *m, const int *v, int *ans, int n) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < n) {
		for (int k = 0; k < n; k++)
			ans[tid] += m[tid*n + k] * v[k];
	}
}

int main() {
	ifstream in("input.txt");
	int dim;
	in >> dim;
	const size_t dSize = sizeof(int)*dim*dim;
	int *m1 = (int*)malloc(dSize), *m2 = (int*)malloc(dSize), *mAns = (int*)malloc(dSize);
	int *v = (int*)malloc(sizeof(int)*dim), *vAns = (int*)malloc(sizeof(int)*dim);

	input(in, dim, m1, m2, mAns, v, vAns);

	int *cuM1 = nullptr, *cuM2 = nullptr, *cuMAns = nullptr, *cuV = nullptr, *cuVAns = nullptr;

	errorHandler(cudaMalloc(&cuM1, dSize), "cudaMalloc: cuM1");
	errorHandler(cudaMalloc(&cuM2, dSize), "cudaMalloc: cuM2");
	errorHandler(cudaMalloc(&cuMAns, dSize), "cudaMalloc: cuMAns");
	errorHandler(cudaMalloc(&cuV, sizeof(int)*dim), "cudaMalloc: cuV");
	errorHandler(cudaMalloc(&cuVAns, sizeof(int)*dim), "cudaMalloc: cuVAns");

	errorHandler(cudaMemcpy(cuM1, m1, dSize, cudaMemcpyHostToDevice), "cudaMemcpy: toDevice: cuM1");
	errorHandler(cudaMemcpy(cuM2, m2, dSize, cudaMemcpyHostToDevice), "cudaMemCpy: toDevice: cuM2");
	errorHandler(cudaMemcpy(cuMAns, mAns, dSize, cudaMemcpyHostToDevice), "cudaMemCpy: toDevice: cuMAns");
	errorHandler(cudaMemcpy(cuV, v, sizeof(int)*dim, cudaMemcpyHostToDevice), "cudaMemcpy: toDevice: cuV");
	errorHandler(cudaMemcpy(cuVAns, vAns, sizeof(int)*dim, cudaMemcpyHostToDevice), "cudaMemcpy: toDevice: cuVAns");

	MxMKernel << <blockPerGrid, threadsPerBlock >> > (cuM1, cuM2, cuMAns, dim);
	errorHandler(cudaGetLastError(), "cudaGetLastError");
	errorHandler(cudaMemcpy(mAns, cuMAns, dSize, cudaMemcpyDeviceToHost), "cudaMemcpy: toHost: mAns");
	outputMxM(dim, m1, m2, mAns);

	MxVKernel << <blockPerGrid, threadsPerBlock >> > (cuM1, cuV, cuVAns, dim);
	errorHandler(cudaGetLastError(), "cudaGetLastError");
	errorHandler(cudaMemcpy(vAns, cuVAns, sizeof(int)*dim, cudaMemcpyDeviceToHost), "cudaMemcpy: toHost: vAns");
	outputMxV(dim, m1, v, vAns);

	free(m1), free(m2), free(mAns), free(v), free(vAns);
	errorHandler(cudaFree(cuM1), "cudaFree: cuM1");
	errorHandler(cudaFree(cuM2), "cudaFree: cuM2");
	errorHandler(cudaFree(cuMAns), "cudaFree: cuMAns");
	errorHandler(cudaFree(cuV), "cudaFree: cuV");
	errorHandler(cudaFree(cuVAns), "cudaFree: cuVAns");

	return 0;
}
