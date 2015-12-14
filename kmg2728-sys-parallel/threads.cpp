// threadExamples.cpp 
//
#include <thread>
#include <vector>
#include <mutex>

using namespace std;
#define COUNT 10
static std::mutex barrier;

//Split "mem" into "parts", e.g. if mem = 10 and parts = 4 you will have: 0,2,4,6,10
//if possible the function will split mem into equal chuncks, if not
//the last chunck will be slightly larger
std::vector<int> bounds(int parts, int mem) {
	std::vector<int>bnd;
	int delta = mem / parts;
	int reminder = mem % parts;
	int N1 = 0, N2 = 0;
	bnd.push_back(N1);
	for (int i = 0; i < parts; ++i) {
		N2 = N1 + delta;
		if (i == parts - 1)
			N2 += reminder;
		bnd.push_back(N2);
		N1 = N2;
	}
	return bnd;
}

void dot_product(const int *v1, const int *v2, int &result, int L, int R){
	int localresult = 0;
	for (int i = L; i < R; ++i){
		// traditional race condition follows in the summing
		localresult += v1[i] * v2[i];
	}

	lock_guard<mutex> block_threads_until_finish_this_job(barrier);
	result += localresult;
}


unsigned int mulTiThreaded(int *resultBuf, const int *arg1buf, const int *arg2buf, unsigned int size)
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