#include "Dotters.h"

#include <stdio.h>
#include <vector>
#include <iostream>
#include <sstream>

using namespace std;

string arrToStr(const int *v, unsigned int size)
{
	stringstream ss('{');

	int i = 0;

	for (; i < size - 1; ++i) ss << v[i] << ',';

	ss << v[i] << '}';

	return ss.str();
}


int main()
{
	const unsigned int size = 1024;
    __declspec(align(16)) const int a[size] = {1, 2, 3, 4, 5, 6, 7, 8};
    __declspec(align(16)) const int b[size] = {10, 10, 10, 10, 10, 10, 10, 10 };
	__declspec(align(16)) int c[size] = { 0 };
	unsigned int sum;

	//cout << "Multithreaded:\n";
	sum = mulTiThreaded(c, a, b, size);
	//cout << arrToStr(a, size) << " dot " << arrToStr(b, size) << " = " << arrToStr(c, size) << "\nSum: " << sum << "\n\n";


	//cout << "SIMD:\n";
	sum = mulSIMD(c, a, b, size);
	//cout << arrToStr(a, size) << " dot " << arrToStr(b, size) << " = " << arrToStr(c, size) << "\nSum: " << sum << "\n\n";


	//cout << "Multithreaded SIMD:\n";
	sum = mulTiThreadedSIMD(c, a, b, size);
	//out << arrToStr(a, size) << " dot " << arrToStr(b, size) << " = " << arrToStr(c, size) << "\nSum: " << sum << "\n\n";


	//out << "Parallel For:\n";
	sum = mulParallelFor(c, a, b, size);
	//out << arrToStr(a, size) << " dot " << arrToStr(b, size) << " = " << arrToStr(c, size) << "\nSum: " << sum << "\n\n";


	//out << "Parallel For SIMD:\n";
	sum = mulParallelForSIMD(c, a, b, size);
	//out << arrToStr(a, size) << " dot " << arrToStr(b, size) << " = " << arrToStr(c, size) << "\nSum: " << sum << "\n\n";


	//out << "CUDA:\n";
	sum = mulWithCudaHostSum(c, a, b, size);
	//out << arrToStr(a, size) << " dot " << arrToStr(b, size) << " = " << arrToStr(c, size) << "\nSum: " << sum << "\n\n";


	/*
	cout << "Press any key\n";
	getchar();
	*/

    return 0;
}