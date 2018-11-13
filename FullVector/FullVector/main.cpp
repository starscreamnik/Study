#include <iostream>
#include <fstream>
#include "fullvector.h"

using namespace std;

void consistentCalculations(const FullVector& a, const FullVector& b, const FullVector& c) {
	cout << "Scalar Mult: " << a.scalarMult(b) << endl;
	cout << "Vector Mult: " << a.vectorMult(b) << endl;
	cout << "Mixed Mult: " << a.mixedMult(b, c) << endl;
}

void parallelCalculations(const FullVector& a, const FullVector& b, const FullVector& c) {
	cout << "Cuda Scalar Mult: " << a.cuScalarMult(b) << endl;
	cout << "Cuda Vector Mult: " << a.cuVectorMult(b) << endl;
	cout << "Cuda Mixed Mult: " << a.cuMixedMult(b, c) << endl;
}

int main() {
	ifstream in("input.txt");
	ofstream out("log.txt");
	int veclen;
	in >> veclen;
	FullVector a(in, out, veclen), b(in, out, veclen), c(in, out, veclen);
	FullVector d = c;
	c = c;

	consistentCalculations(a, b, c);
	parallelCalculations(a, b, c);

	return 0;
}