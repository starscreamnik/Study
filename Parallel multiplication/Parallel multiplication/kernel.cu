#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <cuda.h>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

void errorHandler(cudaError_t error, const string& msg) {
	if (error != cudaSuccess) {
		cerr << msg << endl;
		exit(0);
	}
}

void input(ifstream& in, const int n, vector<int>& m1, vector<int>& m2, vector<int>& v) {
	for (int i = 0; i < n*n; i++) in >> m1[i];
	for (int i = 0; i < n*n; i++) in >> m2[i];
	for (int i = 0; i < n; i++) in >> v[i];
}

void outputMxM(ofstream& out, const int n, vector<int> a, vector<int> b, vector<int> ans) {
	out << "============Matrix on Matrix=============" << endl;
	out << "Parallel:\n";

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			out << ans[i*n + j] << " ";
		}
		out << endl;
	}

	out << "Consistent:\n";
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			int c = 0;
			for (int k = 0; k < n; k++) {
				c += a[n*i + k] * b[n*k + j];
			}
			out << c << " ";
		}
		out << "\n";
	}
	out << endl;
}

void outputMxV(ofstream& out, const int n, vector<int> m, vector<int> v, vector<int> ans) {
	out << "============Matrix on Vector=============" << endl;
	out << "Parallel:\n";
	for (int i = 0; i < n; i++) {
		out << ans[i] << " ";
	}
	out << endl;

	out << "Consistent:\n";
	for (int i = 0; i < n; i++) {
		int c = 0;
		for (int j = 0; j < n; j++) {
			c += m[i*n + j] * v[j];
		}
		out << c << " ";
	}
	out << endl;
}

__global__ void MxMKernel(const int *m1, const int *m2, int *ans, int n) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	int i = tid, j = tid;
	if (tid < n*n) {
		if (tid >= n) {
			j %= n;
			i /= n;
		}
		else i = 0;
		for (int k = 0; k < n; k++)
			ans[tid] += m1[i*n + k] * m2[k*n + j];
	}
}

__global__ void MxMKernelS(const int *m1, const int *m2, int *ans, int n) {
	int xInd = blockIdx.x * blockDim.x + threadIdx.x;
	int yInd = blockIdx.y * blockDim.y + threadIdx.y;

}

__global__ void MxVKernel(const int *m, const int *v, int *ans, int n) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < n) {
		for (int k = 0; k < n; k++)
			ans[tid] += m[k*n + tid] * v[k];
	}
}
__global__ void MxVKernelnon(const int *m, const int *v, int *ans, int n) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < n) {
		for (int k = 0; k < n; k++)
			ans[tid] += m[tid*n + k] * v[k];
	}
}

int main() {
	ifstream in("input.txt");
	ofstream out("output.txt");
	cudaEvent_t start, stop;
	errorHandler(cudaEventCreate(&start), "");
	errorHandler(cudaEventCreate(&stop), "");
	float time;

	int n;
	in >> n;
	const size_t d1Size = sizeof(int)*n;
	const size_t d2Size = d1Size*n;
	vector<int> m1(n*n, 0), m2(n*n, 0), mAns(n*n, 0),
				v(n, 0), vAns(n, 0);

	input(in, n, m1, m2, v);

	vector<int> trM1(n*n, 0);
	for (int i = 0; i < n; i++) 
		for (int j = 0; j < n; j++) 
			trM1[i*n + j] = m1[j*n + i];

	int *cuM1 = nullptr, *cuM2 = nullptr, *cuMAns = nullptr, *cuV = nullptr, *cuVAns = nullptr;

	int *cuTrM1 = nullptr;
	errorHandler(cudaMalloc(&cuTrM1, d2Size), "cudaMalloc: cuTrM1");
	errorHandler(cudaMemcpy(cuTrM1, trM1.data(), d2Size, cudaMemcpyHostToDevice), "cudaMemcpy: toDevice: cuTrM1");

	errorHandler(cudaMalloc(&cuM1, d2Size), "cudaMalloc: cuM1");
	errorHandler(cudaMalloc(&cuM2, d2Size), "cudaMalloc: cuM2");
	errorHandler(cudaMalloc(&cuMAns, d2Size), "cudaMalloc: cuMAns");
	errorHandler(cudaMalloc(&cuV, d1Size), "cudaMalloc: cuV");
	errorHandler(cudaMalloc(&cuVAns, d1Size), "cudaMalloc: cuVAns");

	errorHandler(cudaMemcpy(cuM1, m1.data(), d2Size, cudaMemcpyHostToDevice), "cudaMemcpy: toDevice: cuM1");
	errorHandler(cudaMemcpy(cuM2, m2.data(), d2Size, cudaMemcpyHostToDevice), "cudaMemCpy: toDevice: cuM2");
	errorHandler(cudaMemcpy(cuMAns, mAns.data(), d2Size, cudaMemcpyHostToDevice), "cudaMemCpy: toDevice: cuMAns");
	errorHandler(cudaMemcpy(cuV, v.data(), d1Size, cudaMemcpyHostToDevice), "cudaMemcpy: toDevice: cuV");
	errorHandler(cudaMemcpy(cuVAns, vAns.data(), d1Size, cudaMemcpyHostToDevice), "cudaMemcpy: toDevice: cuVAns");

	const int threadsPerBlock = 256;
	const int blockPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
	dim3 blockSize = dim3(threadsPerBlock, threadsPerBlock, 1);
	dim3 gridSize = dim3(blockPerGrid, blockPerGrid, 1);

	errorHandler(cudaEventRecord(start), "");
	MxVKernelnon << <blockPerGrid, threadsPerBlock >> > (cuM1, cuV, cuVAns, n);
	errorHandler(cudaGetLastError(), "cudaGetLastError");
	errorHandler(cudaDeviceSynchronize(), "sync");
	errorHandler(cudaMemcpy(vAns.data(), cuVAns, d1Size, cudaMemcpyDeviceToHost), "cudaMemcpy: toHost: vAns");
	outputMxV(out, n, m1, v, vAns);
	errorHandler(cudaEventRecord(stop), "" );
	errorHandler(cudaEventSynchronize(stop), "");
	errorHandler(cudaEventElapsedTime(&time, start, stop), "");
	printf("MxV non-coalesced time X %f\n", time);

	vector<int> Empty(n, 0);
	errorHandler(cudaMemcpy(cuVAns, Empty.data(), d1Size, cudaMemcpyHostToDevice), "cudaMemcpy: toDevice: cuVAns");

	errorHandler(cudaEventRecord(start), "");
	MxVKernel << <blockPerGrid, threadsPerBlock >> > (cuTrM1, cuV, cuVAns, n);
	errorHandler(cudaGetLastError(), "cudaGetLastError");
	errorHandler(cudaDeviceSynchronize(), "sync");
	errorHandler(cudaMemcpy(vAns.data(), cuVAns, d1Size, cudaMemcpyDeviceToHost), "cudaMemcpy: toHost: vAns");
	outputMxV(out, n, m1, v, vAns);
	errorHandler(cudaEventRecord(stop), "");
	errorHandler(cudaEventSynchronize(stop), "");
	errorHandler(cudaEventElapsedTime(&time, start, stop), "");
	printf("MxV coalesced time X %f\n", time);

	//errorHandler(cudaEventRecord(start), "");
	//MxMKernel << <gridSize, blockSize >> > (cuM1, cuM2, cuMAns, n);
	//
	//errorHandler(cudaGetLastError(), "cudaGetLastError");
	//errorHandler(cudaDeviceSynchronize(), "sync");
	//errorHandler(cudaMemcpy(vAns.data(), cuVAns, d1Size, cudaMemcpyDeviceToHost), "cudaMemcpy: toHost: vAns");
	//outputMxM(out, n, m1, v, vAns);
	//errorHandler(cudaEventRecord(stop), "");
	//errorHandler(cudaEventSynchronize(stop), "");
	//errorHandler(cudaEventElapsedTime(&time, start, stop), "");
	//printf("coalesced time X %f\n", time);


	//MxMKernel << <blockPerGrid, threadsPerBlock >> > (cuM1, cuM2, cuMAns, n);
	//errorHandler(cudaGetLastError(), "cudaGetLastError");
	//errorHandler(cudaDeviceSynchronize(), "sync");
	//errorHandler(cudaMemcpy(mAns, cuMAns, d2Size, cudaMemcpyDeviceToHost), "cudaMemcpy: toHost: mAns");
	//outputMxM(out, n, m1, m2, mAns);

	errorHandler(cudaFree(cuTrM1), "cudaFree: cuTrM1");
	errorHandler(cudaFree(cuM1), "cudaFree: cuM1");
	errorHandler(cudaFree(cuM2), "cudaFree: cuM2");
	errorHandler(cudaFree(cuMAns), "cudaFree: cuMAns");
	errorHandler(cudaFree(cuV), "cudaFree: cuV");
	errorHandler(cudaFree(cuVAns), "cudaFree: cuVAns");

	return 0;
}
