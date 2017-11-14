#include <cstdio>
#include <cuda_fp16.h>
#include <assert.h>
#include "fp16_conversion.h"   // host function for half conversion

#define TWO_PI             6.28318530717959f
#define TWO_PI_H	__float2half(6.28318530717959f)
//#define TWO_PI             6.28

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

__global__
void myTest(int n, float a, const float *x, half *y)
{
	int gid = threadIdx.x + blockDim.x * blockIdx.x;

	half a_half = __float2half(a);
	half2 a2 = __halves2half2(a_half, a_half);

	half one_half = __float2half(1.f);
	half2 one2 = __halves2half2( __float2half(1.f), __float2half(2.f) );

	// half2 add
	// result.low = 1
	// result.high = 2
	half2 result = __hadd2(a2, one2);

	if(gid<n) {
		y[gid] = __low2half(result);  // 1 + 1
		//y[gid] = __high2half(result);  // 1 + 2
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
  printf("a = %f\n", a);

  float *x;
  checkCuda(cudaMallocManaged(&x, n * sizeof(float)));

  half *y;
  checkCuda(cudaMallocManaged(&y, n * sizeof(half)));
  
  for (int i = 0; i < n; i++) {
    x[i] = 1.0f;
    y[i] = approx_float_to_half(2.f);
  }


  const int blockSize = 32;
  const int nBlocks = (n + blockSize - 1) / blockSize;

  myTest<<<nBlocks, blockSize>>>(n, a, x, y);

  // must wait for kernel to finish before CPU accesses
  checkCuda(cudaDeviceSynchronize());
  
  for (int i = 0; i < n; i++)
  	printf("%f\n", half_to_float(y[i]));


  return 0;
}
