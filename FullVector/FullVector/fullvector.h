#ifndef __FULLVECTOR_H__
#define __FULLVECTOR_H__

#include <iostream>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <string>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

typedef vector<vector<int>> dim2;

class FullVector {
public:
	FullVector(ifstream& in, ofstream& logFile, int n);
	FullVector(int n, ofstream& logFile);
	FullVector(const FullVector& obj);
	FullVector& operator=(const FullVector& obj);
	~FullVector();

	int getLen()const;
	vector<int> getVector()const;
	ofstream* getLogFile()const;

	void upgradeVector(const int* cuVec);

	dim2 getMinor(const dim2& det, int n, int ik, int jk)const;
	int determinant(const dim2& det, int n)const;

	int scalarMult(const FullVector& v1)const;
	int vectorMult(const FullVector& v1)const;
	int mixedMult(const FullVector& v1, const FullVector& v2)const;
	
	int cuScalarMult(const FullVector& objVec1)const;
	int cuVectorMult(const FullVector& objVec1)const;
	int cuMixedMult(const FullVector& objVec1, const FullVector& objVec2)const;

private: 
	int len;
	vector<int> vec;
	ofstream* log;
};

#endif //__FULLVECTOR_H__