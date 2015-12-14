#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Dotters.h"

#include <stdio.h>
#include <exception>
#include <numeric>

#define COOLDAE(x) { if (x != cudaSuccess) throw std::exception(#x); }
#define COOLDAEG(x) { cudaError_t status; status = x; if (status != cudaSuccess) throw std::exception(cudaGetErrorString(status)); }
#define COOLDAG(x, l) { if (x != cudaSuccess) { fprintf(stderr, #x); goto l; }}

__global__ void mulKernel(int *productbuf, const int *arg1buf, const int *arg2buf)
{
	int i = threadIdx.x;
	productbuf[i] = arg1buf[i] * arg2buf[i];
}

typedef void* cudaBuf;

void cudaSetup(cudaBuf *a, cudaBuf *b, cudaBuf *c, unsigned int size)
{
	COOLDAE(cudaSetDevice(0));
	COOLDAE(cudaMalloc(a, size * sizeof(int)));
	COOLDAE(cudaMalloc(b, size * sizeof(int)));
	COOLDAE(cudaMalloc(c, size * sizeof(int)));
}

void cudaTeardown(cudaBuf a, cudaBuf b, cudaBuf c)
{
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
}

void cudaTransfer(cudaBuf cudabuf, const int *hostdata, unsigned int size)
{
	COOLDAE(cudaMemcpy(cudabuf, hostdata, size * sizeof(int), cudaMemcpyHostToDevice));
}

void cudaGetBack(int *hostdata, cudaBuf cudabuf, unsigned int size)
{
	COOLDAE(cudaMemcpy(hostdata, cudabuf, size * sizeof(int), cudaMemcpyDeviceToHost));
}


unsigned int mulWithCudaHostSum(int *c, const int *a, const int *b, unsigned int size)
{
	int *deva = nullptr;
	int *devb = nullptr;
	int *devc = nullptr;

	try {

		cudaSetup((void**)&deva, (void**)&devb, (void**)&devc, size);
		cudaTransfer(deva, a, size);
		cudaTransfer(devb, b, size);

		mulKernel << <1, size >> >(devc, deva, devb);

		// Check for any errors launching the kernel
		COOLDAEG(cudaGetLastError());

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		COOLDAE(cudaDeviceSynchronize());

		cudaGetBack(c, devc, size);

		auto sum = std::accumulate(c, c + size, 0);

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		COOLDAE(cudaDeviceReset());

		return sum;

	} catch (std::exception e) {
		cudaFree(devc);
		cudaFree(devb);
		cudaFree(deva);

		throw e;
	}
}