#ifndef __FULLVECTOR_H__
#define __FULLVECTOR_H__

#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <string>


using namespace std;

typedef vector<vector<int>> dim2;

class FullVector {
public:
	FullVector(ifstream& in, int n);
	FullVector(int n);
	~FullVector();

	int getLen()const;
	size_t getSize()const;
	vector<int> getVector()const;
	void upgradeVector();

	void operator=(const FullVector& obj);
	dim2 getMinor(const dim2& det, int n, int ik, int jk)const;
	int determinant(const dim2& det, int n)const;

	int scalarMult(const FullVector& v1)const;
	int vectorMult(const FullVector& v1)const;
	int mixedMult(const FullVector& v1, const FullVector& v2)const;
	
	int cuScalarMult(FullVector& objVec1);
	int cuVectorMult(FullVector& objVec1);
	int cuMixedMult(FullVector& objVec1, FullVector& objVec2);

private: 
	int* getCuVector();
	int len;
	const size_t size; 
	vector<int> vec;
	int* cuVec;
};

#endif //__FULLVECTOR_H__