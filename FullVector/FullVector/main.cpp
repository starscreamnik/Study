#include <iostream>
#include <fstream>
#include "fullvector.h"

using namespace std;

void consistentCalculations(const FullVector& a, const FullVector& b, const FullVector& c) {
	cout << "Scalar Mult: " << a.scalarMult(b) << endl;
	cout << "Vector Mult: " << a.vectorMult(b) << endl;
	cout << "Mixed Mult: " << a.mixedMult(b, c) << endl;
}

void parallelCalculations(/*FullVector& a, FullVector& b, FullVector& c*/) {
	cout << "Cuda Scalar Mult: " << a.cuScalarMult(b) << endl;
	cout << "Cuda Vector Mult: " << a.cuVectorMult(b) << endl;
	cout << "Cuda Mixed Mult: " << a.cuMixedMult(b, c) << endl;
}

int main() {
	ifstream in("input.txt");
	int veclen;
	in >> veclen;
	FullVector a(in, veclen), b(in, veclen), c(in, veclen);

	consistentCalculations(a, b, c);
	parallelCalculations(a, b, c);

	return 0;
}