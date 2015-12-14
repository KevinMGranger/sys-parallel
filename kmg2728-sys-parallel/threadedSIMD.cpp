#include "Dotters.h"
#include <numeric>
#include <thread>
#include <mutex>

using namespace std;
static std::mutex barrier;

static void dot_product(const int *arg1buf, const int *arg2buf, int &result, int L, int R)
{
	int localresult = 0;

	__declspec(align(16)) int results[4] = { 0 };

	for (int i = L; i < R; i += 4) {
		__m128i a = _mm_load_si128((__m128i const*)(arg1buf + i));

		__m128i b = _mm_load_si128((__m128i const*)(arg2buf + i));

		__m128i prod = _mm_mullo_epi32(a, b);

		_mm_store_si128((__m128i *)(results), prod);

		localresult = std::accumulate(results, results + 4, localresult);
	}

	lock_guard<mutex> resultWriter(barrier);
	result += localresult;
}

unsigned int mulTiThreadedSIMD(int *resultBuf, const int *arg1buf, const int *arg2buf, unsigned int size)
{
	int nr_threads = 2;
	int result = 0;
	std::vector<std::thread> threads;

	//Split nr_elements into nr_threads parts
	std::vector<int> limits = bounds(nr_threads, size);

	//Launch nr_threads threads:
	for (int i = 0; i < nr_threads; ++i) {
		threads.push_back(std::thread(dot_product, arg1buf, arg2buf, std::ref(result), limits[i], limits[i + 1]));
	}


	//Join the threads with the main thread
	for (auto &t : threads){
		t.join();
	}

	return result;
}
