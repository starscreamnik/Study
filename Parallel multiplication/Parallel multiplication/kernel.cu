#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <cuda.h>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

const int threadsPerBlock = 32;

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

void checkMxV(const int n, const vector<int>& m, const vector<int>& v, const vector<int>& ans) {
	for (int i = 0; i < n; i++) {
		int c = 0;
		for (int j = 0; j < n; j++) {
			c += m[i*n + j] * v[j];
		}
		if (ans[i] != c) {
			cout << "false" << endl;
			cout << c <<":" << ans[i] << endl;
			return;
		}
	}
	cout << "true" << endl;
}

void checkMxM(const int n, const vector<int>& a, const vector<int>& b, const vector<int>& ans) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			int c = 0;
			for (int k = 0; k < n; k++) {
				c += a[i*n + k] * b[n*k + j];
			}
			if (c != ans[i*n + j]) {
				cout << "false" << " (i,j) "<<i<<"," <<j <<" " ;
				cout << c << ":" << ans[i*n + j] << endl;
				return;
			}
		}
	}
	cout << "true" << endl;
}

__global__ void MxV(const int *m, const int *v, int *ans, int n) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n) {
		int tmp = 0;
		for (int k = 0; k < n; k++)
			tmp += m[idx*n + k] * v[k];
		ans[idx] = tmp;
	}
}

__global__ void MxV_coalessed(const int *m, const int *v, int *ans, int n) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while (idx < n) {
		int tmp = 0;
		for (int k = 0; k < n; k++)
			tmp += m[k*n + idx] * v[k];
		ans[idx] = tmp;
		idx += gridDim.x * blockDim.x;
	}
}

__global__ void MxV_shared_slow(const int *m, const int *v, int *ans, int n) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ int cache[4];
	int cacheIndex = threadIdx.x;

	if (blockIdx.x < n) {
		cache[cacheIndex] = m[idx] * v[blockIdx.x];
		__syncthreads();

		int i = blockDim.x / 2;
		while(i != 0) {
			if (cacheIndex < i) {
				cache[cacheIndex] += cache[cacheIndex + i];
				__syncthreads();
			}
			i /= 2;
		}
		if (cacheIndex == 0) {
			ans[blockIdx.x] = cache[0];
		}
	}
}

__global__ void MxV_shared(const int *m, const int *v, int *ans, int n) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int round = gridDim.x;
	int sum = 0;
	__shared__ int cache[threadsPerBlock];

	for (int cycle = 0; cycle < round; cycle++) {
		cache[threadIdx.x] = v[cycle*threadsPerBlock + threadIdx.x];
		__syncthreads();

		for (int i = 0; i < threadsPerBlock; i++) {
			if (cycle*threadsPerBlock + i < n)
				sum += m[(cycle*threadsPerBlock + i)*n + idx] * cache[i];
		}
		__syncthreads();
	}
	ans[idx] = sum;
}

__global__ void MxM_pitched(const int n, const int pitchM1, const int pitchM2,  const int *m1, const int *m2, int *ans) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	if (idx < n && idy < n) {
		int val = 0;
		for (int k = 0; k < n; k++) 
			val += m1[idx*pitchM1 + k] * m2[k*pitchM2 + idy];
		ans[idx*n + idy] = val;
	}
}

__global__ void MxM_shared(int *m1, int *m2, int *ans, const int pitchM1, const int pitchM2, int n) {
	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;
	int aBegin = pitchM1 * threadsPerBlock * by;
	int aEnd = aBegin + n - 1;
	int bBegin = threadsPerBlock * bx;
	int aStep = threadsPerBlock, bStep = threadsPerBlock * pitchM2;
	int sum = 0;
	
		for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep) {
			__shared__ int as[threadsPerBlock][threadsPerBlock];	// create memory
			__shared__ int bs[threadsPerBlock][threadsPerBlock];
			as[ty][tx] = m1[ia + pitchM1 * ty + tx];				//read from dram to shared
			bs[ty][tx] = m2[ib + pitchM2 * ty + tx];
			__syncthreads();							// Synchronize to make sure the matrices are loaded

			for (int k = 0; k < threadsPerBlock; k++)				//solve
				sum += as[ty][k] * bs[k][tx];
			__syncthreads();							// Synchronize to make sure submatrices not needed
		}

		ans[n * threadsPerBlock * by + threadsPerBlock * bx + n * ty + tx] = sum;
}

int main() {
	ifstream in("input.txt");
	ofstream out("output.txt");

	int n;
	in >> n;
	const size_t d1Size = sizeof(int)*n;
	const size_t d2Size = d1Size*n;
	vector<int> m1(n*n, 0), m2(n*n, 0), mAns(n*n, 0),
			    v(n, 0), vAns(n, 0);
	vector<int> Empty(n, 0);

	input(in, n, m1, m2, v);

	vector<int> trM1(n*n, 0);
	for (int i = 0; i < n; i++) 
		for (int j = 0; j < n; j++) 
			trM1[i*n + j] = m1[j*n + i];

	int *cuM1 = nullptr, *cuM2 = nullptr, *cuMAns = nullptr, *cuV = nullptr, *cuVAns = nullptr;

	int *cuTrM1 = nullptr;
	errorHandler(cudaMalloc(&cuTrM1, d2Size), "cudaMalloc: cuTrM1");
	errorHandler(cudaMemcpy(cuTrM1, trM1.data(), d2Size, cudaMemcpyHostToDevice), "cudaMemcpy: toDevice: cuTrM1");

	//cudaMalloc
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

	//set blockSize and gridSize
	const int blockPerGrid = (n + threadsPerBlock - 1 )/ threadsPerBlock;
	dim3 blockSize = dim3(threadsPerBlock, threadsPerBlock, 1);
	dim3 gridSize = dim3(blockPerGrid, blockPerGrid, 1);

	cudaEvent_t mStart, mStop;
	cudaEventCreate(&mStart);
	cudaEventCreate(&mStop);
	float elapsed;

	//Matrix on Vector multiplication is simple
	cudaEventRecord(mStart);
	cout << "============Matrix on Vector=============" << endl;
	MxV << <blockPerGrid, threadsPerBlock >> > (cuM1, cuV, cuVAns, n);
	errorHandler(cudaGetLastError(), "cudaGetLastError after 'MxV' call");
	errorHandler(cudaDeviceSynchronize(), "DeviceSynchronize");
	errorHandler(cudaEventRecord(mStop), "Record: stop");
	errorHandler(cudaEventSynchronize(mStop), "EventSynchronize: stop");
	errorHandler(cudaEventElapsedTime(&elapsed, mStart, mStop), "ElapsedTime");

	errorHandler(cudaMemcpy(vAns.data(), cuVAns, d1Size, cudaMemcpyDeviceToHost), "cudaMemcpy: toHost: vAns");
	checkMxV(n, m1, v, vAns);
	cout << "simple: " << elapsed << "\n\n";

	errorHandler(cudaMemcpy(cuVAns, Empty.data(), d1Size, cudaMemcpyHostToDevice), "cudaMemcpy: toDevice: cuVAns");

	//Matrix on Vector multiplication with coalessed
	cudaEventRecord(mStart);
	MxV_coalessed << <blockPerGrid, threadsPerBlock >> > (cuTrM1, cuV, cuVAns, n);
	errorHandler(cudaGetLastError(), "cudaGetLastError after 'MxV_coalessed' call");
	errorHandler(cudaDeviceSynchronize(), "DeviceSynchronize");
	errorHandler(cudaEventRecord(mStop), "Record: stop");
	errorHandler(cudaEventSynchronize(mStop), "EventSynchronize: stop");
	errorHandler(cudaEventElapsedTime(&elapsed, mStart, mStop), "ElapsedTime");

	errorHandler(cudaMemcpy(vAns.data(), cuVAns, d1Size, cudaMemcpyDeviceToHost), "cudaMemcpy: toHost: vAns");
	checkMxV(n, m1, v, vAns);
	cout << "coalessed: " << elapsed << "\n\n";

	errorHandler(cudaMemcpy(cuVAns, Empty.data(), d1Size, cudaMemcpyHostToDevice), "cudaMemcpy: toDevice: cuVAns");

	//Matrix on Vector multiplication with shared memory 
	cudaEventRecord(mStart);
	MxV_shared << <blockPerGrid, threadsPerBlock >> > (cuM2, cuV, cuVAns, n);
	errorHandler(cudaGetLastError(), "cudaGetLastError after 'MxV_shared' call");
	errorHandler(cudaDeviceSynchronize(), "error: DeviceSynchronize");
	errorHandler(cudaEventRecord(mStop), "error: Record: mstop");
	errorHandler(cudaEventSynchronize(mStop), "error: EventSynchronize: mstop");
	errorHandler(cudaEventElapsedTime(&elapsed, mStart, mStop), "ElapsedTime");

	errorHandler(cudaMemcpy(vAns.data(), cuVAns, d1Size, cudaMemcpyDeviceToHost), "cudaMemcpy: toHost: vAns");
	checkMxV(n, m2, v, vAns);
	cout << "shared memory: " << elapsed << "\n\n";


	size_t pitch_mat_1;
	size_t pitch_mat_2;
	errorHandler(cudaMallocPitch((void**)&cuM1, &pitch_mat_1, n * sizeof(int), n), "mallocPitch");  // width in bytes by height
	errorHandler(cudaMallocPitch((void**)&cuM2, &pitch_mat_2, n * sizeof(int), n), "mallocPitch");

	errorHandler(cudaMemcpy2D(cuM1, pitch_mat_1, &m1.front(), n * sizeof(int), n * sizeof(float), n, cudaMemcpyHostToDevice), "cudaMemcpy2D");
	errorHandler(cudaMemcpy2D(cuM2, pitch_mat_2, &m2.front(), n * sizeof(int), n * sizeof(float), n, cudaMemcpyHostToDevice), "cudaMemcpy2D");

	//Matrix on Matrix multiplication pitched
	cudaEventRecord(mStart);
	cout << "============Matrix on Matrix=============" << endl;
	MxM_pitched << <gridSize, blockSize>> > (n, pitch_mat_1 / sizeof(int), pitch_mat_2 / sizeof(int), cuM1, cuM2, cuMAns);
	errorHandler(cudaGetLastError(), "cudaGetLastError after 'MxM_pitched' call");
	errorHandler(cudaDeviceSynchronize(), "DeviceSynchronize");
	errorHandler(cudaEventRecord(mStop), "Record: stop");
	errorHandler(cudaEventSynchronize(mStop), "EventSynchronize: stop");
	errorHandler(cudaEventElapsedTime(&elapsed, mStart, mStop), "ElapsedTime");
	errorHandler(cudaMemcpy(mAns.data(), cuMAns, d2Size, cudaMemcpyDeviceToHost), "cudaMemcpy: toHost: mAns");
	
	checkMxM(n, m1, m2, mAns);
	cout << "pitched: " << elapsed << "\n\n";

	mAns.assign(n*n, 0);
	errorHandler(cudaMemcpy(cuMAns, mAns.data(), d2Size, cudaMemcpyHostToDevice), "cudaMemcpy: toDevice: cuMAns");

	//Matrix on Matrix multiplication with shared memory
	cudaEventRecord(mStart);
	MxM_shared << <gridSize, blockSize >> > (cuM1, cuM2, cuMAns, pitch_mat_1 / sizeof(int), pitch_mat_2 / sizeof(int), n);
	errorHandler(cudaGetLastError(), "cudaGetLastError after 'MxM_shared' call");
	errorHandler(cudaDeviceSynchronize(), "DeviceSynchronize");
	errorHandler(cudaEventRecord(mStop), "Record: stop");
	errorHandler(cudaEventSynchronize(mStop), "EventSynchronize: stop");
	errorHandler(cudaEventElapsedTime(&elapsed, mStart, mStop), "ElapsedTime");
	errorHandler(cudaMemcpy(mAns.data(), cuMAns, d2Size, cudaMemcpyDeviceToHost), "cudaMemcpy: toHost: mAns");

	checkMxM(n, m1, m2, mAns);
	cout << "shared memory: " << elapsed << endl;

	cudaEventDestroy(mStart);
	cudaEventDestroy(mStop);
	errorHandler(cudaFree(cuTrM1), "cudaFree: cuTrM1");
	errorHandler(cudaFree(cuM1), "cudaFree: cuM1");
	errorHandler(cudaFree(cuM2), "cudaFree: cuM2");
	errorHandler(cudaFree(cuMAns), "cudaFree: cuMAns");
	errorHandler(cudaFree(cuV), "cudaFree: cuV");
	errorHandler(cudaFree(cuVAns), "cudaFree: cuVAns");

	return 0;
}
