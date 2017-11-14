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

__global__ void fp32_cmp(int n, float a, float *x, half *y, uint *start_time, uint *end_time)
{
	int gid = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int start_time1;
	unsigned int end_time1;
	float htime;

	__syncthreads();

	start_time1 = clock();
	htime = (a>=0.f);
	end_time1 = clock();

	__syncthreads();

	x[gid] = htime;

	__syncthreads();

	if (gid == 0) {
		start_time[0] = start_time1;
		end_time[0] = end_time1;
	}
}


/*
__global__ void fp16_cmp(int n, float a, float *x, half *y, uint *start_time, uint *end_time)
{
	int gid = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int start_time1;
	unsigned int end_time1;

	half a_half = __float2half(a);
	half zero_half = __float2half(0.f);
	half result;

	__syncthreads();

	start_time1 = clock();
	result = __float2half(float(__hge(a_half, zero_half)));
	end_time1 = clock();

	__syncthreads();

	y[gid] = result; 

	__syncthreads();

	if (gid == 0) {
		start_time[0] = start_time1;
		end_time[0] = end_time1;
	}
}
*/

/*
__global__ void fp32_floor(int n, float a, float *x, half *y, uint *start_time, uint *end_time)
{
	int gid = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int start_time1;
	unsigned int end_time1;
	float htime;

	__syncthreads();

	start_time1 = clock();
	htime = floorf(a);
	end_time1 = clock();

	__syncthreads();


	if (gid == 0) {
		start_time[0] = start_time1;
		end_time[0] = end_time1;
	}

	__syncthreads();

	x[gid] = htime;

}
*/

/*
__global__ void fp16_floor(int n, float a, float *x, half *y, uint *start_time, uint *end_time)
{
	int gid = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int start_time1;
	unsigned int end_time1;
	half a_half = __float2half(a);
	half htime;

	__syncthreads();

	start_time1 = clock();
	htime = hfloor(a_half);
	end_time1 = clock();

	__syncthreads();


	if (gid == 0) {
		start_time[0] = start_time1;
		end_time[0] = end_time1;
	}

	__syncthreads();

	y[gid] = htime;

}
*/


/*
__global__ void fp32_add(int n, float a, float b, float *x, half *y, uint *start_time, uint *end_time)
{
	int gid = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int start_time1;
	unsigned int end_time1;
	float v;

	float t1  = a + x[gid];
	float t2  = b + x[gid];

	__syncthreads();

	// inline ptx
	start_time1 = clock();
	//htime = a + b;
	//htime = t1 + t2;
	//t1 = t1 + t2;
	asm volatile (
			"add.f32 %0, %1, %2;\n\t" : "=f"(v) : "f"(t1), "f"(t2)
		     );
	end_time1 = clock();

	__syncthreads();

	if (gid == 0) {
		start_time[0] = start_time1;
		end_time[0] = end_time1;
	}

	__syncthreads();

	x[gid] = v;
}
*/


/*
__global__ void fp16_add(int n, float a, float b,  float *x, half *y, uint *start_time, uint *end_time)
{
	int gid = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int start_time1;
	unsigned int end_time1;
	half a_half = __float2half(a);
	half b_half = __float2half(b);
	half result;

	__syncthreads();

	start_time1 = clock();
	result = __hadd(a_half, b_half);
	end_time1 = clock();

	__syncthreads();


	if (gid == 0) {
		start_time[0] = start_time1;
		end_time[0] = end_time1;
	}

	__syncthreads();

	y[gid] = result;

}
*/

/*
__global__ void fp32_mul(int n, float a, float b, float *x, half *y, uint *start_time, uint *end_time)
{
	int gid = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int start_time1;
	unsigned int end_time1;
	float v;

	float t1  = a + x[gid];
	float t2  = b + x[gid];

	__syncthreads();

	// inline ptx
	start_time1 = clock();
	//v = t1 * t2;
	asm volatile ("mul.f32 %0, %1, %2;\n\t"
	: "=f"(v) : "f"(t1) , "f"(t2));
	end_time1 = clock();

	__syncthreads();

	if (gid == 0) {
		start_time[0] = start_time1;
		end_time[0] = end_time1;
	}

	__syncthreads();

	x[gid] = v;
}
*/

/*
__global__ void fp16_mul(int n, float a, float b,  float *x, half *y, uint *start_time, uint *end_time)
{
	int gid = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int start_time1;
	unsigned int end_time1;
	half a_half = __float2half(a);
	half b_half = __float2half(b);
	half result;

	__syncthreads();

	start_time1 = clock();
	result = __hmul(a_half, b_half);
	end_time1 = clock();

	__syncthreads();


	if (gid == 0) {
		start_time[0] = start_time1;
		end_time[0] = end_time1;
	}

	__syncthreads();

	y[gid] = result;

}
*/

int main(int argc, char** argv) {

	int devid = atoi(argv[1]);
	cudaSetDevice(devid);

	cudaDeviceProp prop;                                                    
	cudaGetDeviceProperties(&prop, devid);                                 
	printf("device %d : %s\n", devid, prop.name);

	const int n = 32;

	const float a = 1.f; 
	const float b = 3.f; 

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



	fp32_cmp<<<1, 32>>>(n, a, x, y, start_time, end_time);
	checkCuda(cudaDeviceSynchronize());
	printf("(fp32 cmp) clocks: %d\n", end_time - start_time);

/*
	fp16_cmp<<<1, 32>>>(n, a, x, y, start_time, end_time);
	checkCuda(cudaDeviceSynchronize());
	printf("(fp16 cmp) clocks: %d\n", end_time - start_time);
*/

/*
	fp32_floor<<<1, 32>>>(n, a, x, y, start_time, end_time);
	checkCuda(cudaDeviceSynchronize());
	printf("(fp32 floor) clocks: %d\n", end_time - start_time);
*/

/*
	fp16_floor<<<1, 32>>>(n, a, x, y, start_time, end_time);
	checkCuda(cudaDeviceSynchronize());
	printf("(fp16 floor) clocks: %d\n", end_time - start_time);
*/


/*
	fp32_add<<<1, 32>>>(n, a, b, x, y, start_time, end_time);
	checkCuda(cudaDeviceSynchronize());
	printf("(fp32 add) clocks: %d\n", end_time - start_time);
*/

/*
	fp16_add<<<1, 32>>>(n, a, b, x, y, start_time, end_time);
	checkCuda(cudaDeviceSynchronize());
	printf("(fp16 add) clocks: %d\n", end_time - start_time);
*/

/*
	fp32_mul<<<1, 32>>>(n, a, b, x, y, start_time, end_time);
	checkCuda(cudaDeviceSynchronize());
	printf("(fp32 mul) clocks: %d\n", end_time - start_time);
*/

/*
	fp16_mul<<<1, 32>>>(n, a, b, x, y, start_time, end_time);
	checkCuda(cudaDeviceSynchronize());
	printf("(fp16 mul) clocks: %d\n", end_time - start_time);
*/

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
