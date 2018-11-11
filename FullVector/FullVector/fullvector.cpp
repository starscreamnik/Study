#include "fullvector.h"
#include "kernel.cuh"

FullVector::FullVector(ifstream& in, int n) :len(n), size(sizeof(int)*n) {
	vec.resize(len, 0);
	for (int i = 0; i < len; i++)
		in >> vec[i];

	doCudaMalloc(cuVec, size, "cudaMalloc");
	doCudaMemcpyHtD(cuVec, vec, size, "cudaMemcpy:toDevice");
}

FullVector::FullVector(int n) :len(n), size(sizeof(int)*n) {
	vec.resize(len, 0);
	doCudaMalloc(cuVec, size, "cudaMalloc");
	doCudaMemcpyHtD(cuVec, vec, size, "cudaMemcpy:toDevice");
}

FullVector::~FullVector() {
	doCudaFree(cuVec, "cudaMalloc");
}

void FullVector::operator=(const FullVector& obj) {
	
}

void FullVector::upgradeVector() {
	doCudaMempyDtH(vec, cuVec, size, "cudaMemcpy:toHost");
}

vector<int> FullVector::getVector()const { return this->vec; }
int FullVector::getLen()const { return this->len; }
size_t FullVector::getSize()const { return this->size; }
int* FullVector::getCuVector() { return this->cuVec; }

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

	dim2 det(len, vector<int>(len, 0));
	vector<int> vec2 = objVec.getVector();

	for (int j = 0; j < len; j++) {
		det[0][j] = 1;
		det[1][j] = this->vec[j];
		det[2][j] = vec2[j];
	}

	return determinant(det, len);
}

int FullVector::mixedMult(const FullVector& objVec2, const FullVector& objVec3)const {
	if (!(this->len == objVec2.getLen() && this->len == objVec3.getLen())) {
		cout << "Vector's length is not equal. Return ";
		return 0;
	}
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

int FullVector::cuScalarMult(FullVector& objVec1) {
	FullVector scalAns(this->len);
	
	callKernelScal(this->cuVec, objVec1->getCuVector(), scalAns->getCuVector(), this->len);
	doCudaDeviceSynchronize();
	scalAns.upgradeVector();

	int ans = 0;
	for (auto it : scalAns.getVector()) ans += it;
	return ans;
}

int FullVector::cuVectorMult(FullVector& objVec1) {
	FullVector vectAns(this->len);

	callKernelVect(this->cuVec, objVec1->getCuVector(), scalAns->getCuVector(), this->len);
	doCudaDeviceSynchronize();
	vectAns.upgradeVector();

	int ans = 0;
	for (auto it : vectAns.getVector()) ans += it;
	return ans;
}

int FullVector::cuMixedMult(FullVector& objVec1, FullVector& objVec2) {
	FullVector mixedAns(this->len);

	callKernelMixed(this->cuVec, objVec1->getCuVector(), objVec2->getCuVector(), mixedAns->getCuVector(), this->len);
	doCudaDeviceSynchronize();
	mixedAns.upgradeVector();

	int ans = 0;
	for (auto it : mixedAns.getVector()) ans += it;
	return ans;
}