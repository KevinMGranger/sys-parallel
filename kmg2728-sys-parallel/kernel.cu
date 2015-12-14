#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Dotters.h"

#include <stdio.h>
#include <stdexcept>
#include <numeric>


#pragma region Error Handling macros
#define COOLDAE(x) { if (x != cudaSuccess) throw std::runtime_error(#x); }
#define COOLDAEG(x) { cudaError_t status; status = x; if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status)); }
#define COOLDAG(x, l) { if (x != cudaSuccess) { fprintf(stderr, #x); goto l; }}
#pragma endregion

static int *deva = nullptr, *devb = nullptr, *devc = nullptr;

__global__ void mulKernel(int *productbuf, const int *arg1buf, const int *arg2buf)
{
	int i = threadIdx.x;
	productbuf[i] = arg1buf[i] * arg2buf[i];
}

typedef void* cudaBuf;

void cudaSetup(unsigned int size)
{
	COOLDAE(cudaSetDevice(0));
	COOLDAE(cudaMalloc((void**)&deva, size * sizeof(int)));
	COOLDAE(cudaMalloc((void**)&devb, size * sizeof(int)));
	COOLDAE(cudaMalloc((void**)&devc, size * sizeof(int)));
}

void cudaTransfer(cudaBuf cudabuf, const int *hostdata, unsigned int size)
{
	COOLDAE(cudaMemcpy(cudabuf, hostdata, size * sizeof(int), cudaMemcpyHostToDevice));
}

void cudaGetBack(int *hostdata, cudaBuf cudabuf, unsigned int size)
{
	COOLDAE(cudaMemcpy(hostdata, cudabuf, size * sizeof(int), cudaMemcpyDeviceToHost));
}

void transferAll(const int *a, const int *b, unsigned int size)
{
		cudaTransfer(deva, a, size);
		cudaTransfer(devb, b, size);
}

unsigned int cudaDo(int *c, unsigned int size)
{
		mulKernel << <1, size >> >(devc, deva, devb);

		// Check for any errors launching the kernel
		COOLDAEG(cudaGetLastError());

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		COOLDAE(cudaDeviceSynchronize());

		cudaGetBack(c, devc, size);

		return std::accumulate(c, c + size, 0);
}

unsigned int mulWithCudaHostSum(int *c, const int *a, const int *b, unsigned int size)
{
	//try {
		cudaSetup(size);
		transferAll(a, b, size);

		auto sum = cudaDo(c, size);

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		COOLDAE(cudaDeviceReset());

		return sum;

		/*
	} catch (std::exception e) {
		cudaFree(devc);
		cudaFree(devb);
		cudaFree(deva);

		printf(e.what());
		throw e;
	}
	*/
}