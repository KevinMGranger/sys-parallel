#include <vector>

unsigned int mulTiThreaded(int *resultBuf, const int *arg1buf, const int *arg2buf, unsigned int size);

unsigned int mulSIMD(int *resultBuf, const int *arg1buf, const int *arg2buf, unsigned int size);

unsigned int mulTiThreadedSIMD(int *resultBuf, const int *arg1buf, const int *arg2buf, unsigned int size);

unsigned int mulParallelFor(int *resultBuf, const int *arg1buf, const int *arg2buf, unsigned int size);

unsigned int mulWithCudaHostSum(int *resultBuf, const int *arg1buf, const int *arg2buf, unsigned int size);

std::vector<int> bounds(int parts, int mem);