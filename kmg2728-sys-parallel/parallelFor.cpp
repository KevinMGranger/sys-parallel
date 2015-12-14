#include "Dotters.h"
#include <Windows.h>
#include <ppl.h>
#include <numeric>

using concurrency::parallel_for;

unsigned int mulParallelFor(int *resultBuf, const int *arg1buf, const int *arg2buf, unsigned int size)
{
	parallel_for(size_t(0), size_t(size), size_t(1), [&](size_t i) {
		resultBuf[i] = arg1buf[i] * arg2buf[i];
	});

	return std::accumulate(resultBuf, resultBuf + size, 0);
}

unsigned int mulParallelForSIMD(int *resultBuf, const int *arg1buf, const int *arg2buf, unsigned int size)
{
	parallel_for(size_t(0), size_t(size), size_t(4), [&](size_t i) {
		__m128i a = _mm_load_si128((__m128i const*)(arg1buf + i));

		__m128i b = _mm_load_si128((__m128i const*)(arg2buf + i));

		__m128i prod = _mm_mullo_epi32(a, b);

		_mm_store_si128((__m128i *)(resultBuf + i), prod);
	});

	return std::accumulate(resultBuf, resultBuf + size, 0);
}