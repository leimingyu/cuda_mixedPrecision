#include <cstdio>
#include <cuda_fp16.h>
#include <assert.h>
#include "fp16_conversion.h"   // host function for half conversion

#define TWO_PI             6.28318530717959f
#define TWO_PI_H	__float2half(6.28318530717959f)

#define LOG 0


inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}

/*
__global__ void fp32_cmp(int n, float a, float *x, half *y, uint *start_time, uint *end_time)
{
	int gid = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int start_time1;
	unsigned int end_time1;
	float htime;

	__syncthreads();

	start_time1 = clock();
	htime = (a>0.f);
	end_time1 = clock();

	__syncthreads();

	x[gid] = htime;

	__syncthreads();

	if (gid == 0) {
		start_time[0] = start_time1;
		end_time[0] = end_time1;
	}
}
*/


__global__ void fp16_cmp(int n, float a, float *x, half *y, uint *start_time, uint *end_time)
{
	int gid = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int start_time1;
	unsigned int end_time1;

	half a_half = __float2half(a);
	half zero_half = __float2half(0.f);
	//bool cmp;
	half result;

	__syncthreads();

	start_time1 = clock();
	//cmp = __hge(a_half, zero_half);	
	result = __float2half(float(__hge(a_half, zero_half)));
	end_time1 = clock();

	__syncthreads();

	//y[gid] = __float2half(float(cmp));
	y[gid] = result; 

	__syncthreads();

	if (gid == 0) {
		start_time[0] = start_time1;
		end_time[0] = end_time1;
	}
}


int main(int argc, char** argv) {

	int devid = atoi(argv[1]);
	cudaSetDevice(devid);

	cudaDeviceProp prop;                                                    
	cudaGetDeviceProperties(&prop, devid);                                 
	printf("device %d : %s\n", devid, prop.name);

	const int n = 32;

	const float a = 1.f; 

	float *x;
	checkCuda(cudaMallocManaged(&x, n * sizeof(float)));

	half *y;
	checkCuda(cudaMallocManaged(&y, n * sizeof(half)));

	uint *start_time, *end_time;
	checkCuda(cudaMallocManaged(&start_time, sizeof(uint)));
	checkCuda(cudaMallocManaged(&end_time, sizeof(uint)));

	for (int i = 0; i < n; i++) {
		x[i] = 0.f;
		y[i] = approx_float_to_half(0.f);
	}


/*
	fp32_cmp<<<1, 32>>>(n, a, x, y, start_time, end_time);
	checkCuda(cudaDeviceSynchronize());
	printf("(fp32 cmp) clocks: %d\n", end_time - start_time);
*/

	fp16_cmp<<<1, 32>>>(n, a, x, y, start_time, end_time);
	checkCuda(cudaDeviceSynchronize());
	printf("(fp16 cmp) clocks: %d\n", end_time - start_time);


#if LOG
	printf("\nx :\n");
	for (int i = 0; i < n; i++)
		printf("%f\n", x[i]);

	printf("\ny :\n");
	for (int i = 0; i < n; i++)
		printf("%f\n", half_to_float(y[i]));
#endif


	return 0;
}
