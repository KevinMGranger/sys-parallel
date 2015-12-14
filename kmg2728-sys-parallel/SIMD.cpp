#include "Dotters.h"
#include <intrin.h>
#include <numeric>

unsigned int mulSIMD(int *resultBuf, const int *arg1buf, const int *arg2buf, unsigned int size)
{
	for (int i = 0; i < size; i += 4) {

		// load 4
		__m128i a = _mm_load_si128((__m128i const*)(arg1buf + i));

		__m128i b = _mm_load_si128((__m128i const*)(arg2buf + i));

		__m128i prod = _mm_mullo_epi32(a, b);

		_mm_store_si128((__m128i *)(resultBuf + i), prod);
	}

	return std::accumulate(resultBuf, resultBuf + size, 0);
}