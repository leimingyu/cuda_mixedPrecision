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
	//int gid = threadIdx.x + blockDim.x * blockIdx.x;

	//half a_half = __float2half(a);
	//y[gid] = __hadd(a_half, __hge(a_half, bool2half(__float2half(0.f)))); 

	half2 zero2 = __floats2half2_rn(0.f, 0.f);
	half2 minusone = __floats2half2_rn(-1.f, -1.f);
	half2 v2    = __floats2half2_rn(2.15f, -3.14f);
	half2 r2;

	printf("v2:  %f %f\n", __low2half(v2),  __high2half(v2));

	// abs
	half2 cmp2 = __hlt2(v2, zero2);
	printf("(compare) %f %f\n", __low2half(cmp2),  __high2half(cmp2));

	r2 = __hmul2(__hadd2(__hadd2(cmp2, cmp2), minusone), v2);
	
	printf("%f %f\n", __low2half(r2),  __high2half(r2));
}

int main(int argc, char** argv) {

  int devid = atoi(argv[1]);
  cudaSetDevice(devid);

  cudaDeviceProp prop;                                                    
  cudaGetDeviceProperties(&prop, devid);                                 
  printf("device %d : %s\n", devid, prop.name);

  const int n = 1;

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


  myTest<<<1, 1>>>(n, a, x, y);

  checkCuda(cudaDeviceSynchronize());
  
/*
  for (int i = 0; i < n; i++)
  	printf("%f\n", half_to_float(y[i]));
*/


  return 0;
}
