#include "fullvector.h"

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blockPerGrid = min(32, (N + threadsPerBlock - 1) / threadsPerBlock);

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

FullVector::FullVector(ifstream& in, ofstream& logFile, int n) :len(n), log(&logFile){
	(*log) << "constructor FullVector(ifstream& in, ofstream& logFile, int n). Received vector: ";
	vec.resize(len, 0);
	for (int i = 0; i < len; i++) 
		in >> vec[i];
	copy(vec.begin(), vec.end(), ostream_iterator<int>((*log), " "));
	(*log) << endl;
	
}

FullVector::FullVector(int n, ofstream& logFile) :len(n), log(&logFile){
	(*log) << "constructor FullVector(int n, ofstream& logFile). Received vector: 0, 0,..." << endl;
	vec.resize(len, 0);
}

FullVector::FullVector(const FullVector& obj) {
	log = obj.getLogFile();
	vec = obj.getVector();
	len = vec.size();
	(*log) << "copy constructor. Received vector: ";
	copy(vec.begin(), vec.end(), ostream_iterator<int>((*log), " "));
	(*log) << endl;
}


FullVector::~FullVector() {
	(*log) << "destructor" << endl;
	log = nullptr;
}

FullVector& FullVector::operator=(const FullVector& obj) {
	(*log) << "operator=" << endl;
	if (this == &obj) return *this;

	log = obj.getLogFile();
	vec = obj.getVector();
	len = vec.size();
	return *this;
}

void FullVector::upgradeVector(const int* cuVec) {
	if (sizeof(cuVec) != sizeof(int)*len) {
		cerr << "upgradeVector: INCORRECT cuVec(const int*)" << endl;
		exit(0);
	}
	(*log) << "upgradeVector" << endl;
	errorHandler(cudaMemcpy(vec.data(), cuVec, sizeof(int)*len, cudaMemcpyDeviceToHost), "cudaMemcpy:toHost");
}

vector<int> FullVector::getVector()const { return vec; }
int FullVector::getLen()const { return len; }
ofstream* FullVector::getLogFile()const { return this->log; }


dim2 FullVector::getMinor(const dim2& det, int n, int row, int col)const {
	dim2 minor(n - 1, vector<int>(n - 1, 0));

	for (int i = 0, im = 0; i < n; i++, im++) {
		if (i == row) {
			if (row < n - 1) i++;
			else continue;
		}
		for (int j = 0, jm = 0; j < n; j++, jm++) {
			if (j == col) {
				if (col < n - 1) j++;
				else continue;
			}
			minor[im][jm] = det[i][j];
		}
	}
	return minor;
}

int FullVector::determinant(const dim2& det, int n)const {
	if (n == 1) return det[0][0];

	int ans = 0;
	for (int j = 0; j < n; j++)
		ans += pow(-1, j) * det[0][j] * determinant(getMinor(det, n, 0, j), n - 1);
	return ans;
}

int FullVector::scalarMult(const FullVector& v1)const {
	if (v1.getLen() != len) {
		cout << "Vector's length is not equal. Return ";
		return 0;
	}
	(*log) << "Consistent Scalar Multiplication" << endl;
	vector<int> v = v1.getVector();
	int ans = 0;
	for (int i = 0; i < len; i++)
		ans += this->vec[i] * v[i];

	return ans;
}

int FullVector::vectorMult(const FullVector& objVec)const {
	if (objVec.getLen() != this->len) {
		cout << "Vector's length is not equal. Return ";
		return 0;
	}
	(*log) << "Consistent Vector Multiplication" << endl;
	dim2 det(len, vector<int>(len, 0));
	vector<int> vec2 = objVec.getVector();

	for (int j = 0; j < len; j++) {
		det[0][j] = 1;
		det[1][j] = vec[j];
		det[2][j] = vec2[j];
	}

	return determinant(det, len);
}

int FullVector::mixedMult(const FullVector& objVec2, const FullVector& objVec3)const {
	if (!(this->len == objVec2.getLen() && this->len == objVec3.getLen())) {
		cout << "Vector's length is not equal. Return ";
		return 0;
	}
	(*log) << "Consistent Mixed Multiplication" << endl;
	dim2 det(len, vector<int>(len, 0));
	vector<int> vec2 = objVec2.getVector();
	vector<int> vec3 = objVec3.getVector();

	for (int j = 0; j < len; j++) {
		det[0][j] = this->vec[j];
		det[1][j] = vec2[j];
		det[2][j] = vec3[j];
	}

	return determinant(det, len);
}

int FullVector::cuScalarMult(const FullVector& objVec1)const {
	(*log) << "Parallel Scalar Multiplication" << endl;
	int *v = nullptr, *v1 = nullptr, *cuAns = nullptr;
	const size_t vSize = len * sizeof(int);
	vector<int> ans(len, 0);

	errorHandler(cudaMalloc(&v, vSize), "cudaMalloc: v");
	errorHandler(cudaMalloc(&v1, vSize), "cudaMalloc: v1");
	errorHandler(cudaMalloc(&cuAns, vSize), "cudaMalloc: cuAns");
	errorHandler(cudaMemcpy(v, vec.data(), vSize, cudaMemcpyHostToDevice), "cudaMemcpy: toDevice: v");
	errorHandler(cudaMemcpy(v1, (objVec1.getVector()).data(), vSize, cudaMemcpyHostToDevice), "cudaMemcpy: toDevice: v1");
	errorHandler(cudaMemcpy(cuAns, ans.data(), vSize, cudaMemcpyHostToDevice), "cudaMemcpy: toDevice: cuAns");

	kernelScalCalc<<<blockPerGrid, threadsPerBlock>>>(v, v1, cuAns, len);
	cudaDeviceSynchronize();
	errorHandler(cudaMemcpy(ans.data(), cuAns, vSize, cudaMemcpyDeviceToHost), "cudaMemcpy: toHost: ans");

	errorHandler(cudaFree(v), "cudaFree: v");
	errorHandler(cudaFree(v1), "cudaFree: v1");
	errorHandler(cudaFree(cuAns), "cudaFree: cuAns");

	int result = 0;
	for (auto it : ans) result += it;
	return result;
}

int FullVector::cuVectorMult(const FullVector& objVec1)const {
	(*log) << "Parallel Vector Multiplication" << endl;
	int *v = nullptr, *v1 = nullptr, *cuAns = nullptr;
	const size_t vSize = len * sizeof(int);
	vector<int> ans(len, 0);

	errorHandler(cudaMalloc(&v, vSize), "cudaMalloc: v");
	errorHandler(cudaMalloc(&v1, vSize), "cudaMalloc: v1");
	errorHandler(cudaMalloc(&cuAns, vSize), "cudaMalloc: cuAns");
	errorHandler(cudaMemcpy(v, vec.data(), vSize, cudaMemcpyHostToDevice), "cudaMemcpy: toDevice: v");
	errorHandler(cudaMemcpy(v1, (objVec1.getVector()).data(), vSize, cudaMemcpyHostToDevice), "cudaMemcpy: toDevice: v1");
	errorHandler(cudaMemcpy(cuAns, ans.data(), vSize, cudaMemcpyHostToDevice), "cudaMemcpy: toDevice: cuAns");

	kernelVectCalc << <blockPerGrid, threadsPerBlock >> >(v, v1, cuAns, len);
	cudaDeviceSynchronize();
	errorHandler(cudaMemcpy(ans.data(), cuAns, vSize, cudaMemcpyDeviceToHost), "cudaMemcpy: toHost: ans");

	errorHandler(cudaFree(v), "cudaFree: v");
	errorHandler(cudaFree(v1), "cudaFree: v1");
	errorHandler(cudaFree(cuAns), "cudaFree: cuAns");

	int result = 0;
	for (auto it : ans) result += it;
	return result;
}

int FullVector::cuMixedMult(const FullVector& objVec1, const FullVector& objVec2)const {
	(*log) << "Parallel Mixed Multiplication" << endl;
	int *v = nullptr, *v1 = nullptr, *v2 = nullptr, *cuAns = nullptr;
	const size_t vSize = len * sizeof(int);
	vector<int> ans(len, 0);

	errorHandler(cudaMalloc(&v, vSize), "cudaMalloc: v");
	errorHandler(cudaMalloc(&v1, vSize), "cudaMalloc: v1");
	errorHandler(cudaMalloc(&v2, vSize), "cudaMalloc: v2");
	errorHandler(cudaMalloc(&cuAns, vSize), "cudaMalloc: cuAns");

	errorHandler(cudaMemcpy(v, vec.data(), vSize, cudaMemcpyHostToDevice), "cudaMemcpy: toDevice: v");
	errorHandler(cudaMemcpy(v1, (objVec1.getVector()).data(), vSize, cudaMemcpyHostToDevice), "cudaMemcpy: toDevice: v1");
	errorHandler(cudaMemcpy(v2, (objVec2.getVector()).data(), vSize, cudaMemcpyHostToDevice), "cudaMemcpy: toDevice: v2");
	errorHandler(cudaMemcpy(cuAns, ans.data(), vSize, cudaMemcpyHostToDevice), "cudaMemcpy: toDevice: cuAns");

	kernelMixedCalc << <blockPerGrid, threadsPerBlock >> >(v, v1, v2, cuAns, len);
	cudaDeviceSynchronize();
	errorHandler(cudaMemcpy(ans.data(), cuAns, vSize, cudaMemcpyDeviceToHost), "cudaMemcpy: toHost: ans");

	errorHandler(cudaFree(v), "cudaFree: v");
	errorHandler(cudaFree(v1), "cudaFree: v1");
	errorHandler(cudaFree(v2), "cudaFree: v2");
	errorHandler(cudaFree(cuAns), "cudaFree: cuAns");

	int result = 0;
	for (auto it : ans) result += it;
	return result;
}