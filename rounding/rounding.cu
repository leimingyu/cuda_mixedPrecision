#include <cstdio>
#include <cuda_fp16.h>
#include <assert.h>
#include "fp16_conversion.h"   // host function for half conversion

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
void myTest(float *x, half *y)
{
    //float v=2.51f;
    //float v=2.5f;
    //float v=-2.5f;
    //float v=-2.51f;

	//float v = 27.999878f;   // 27 , 28
	//float v = 27.999278f; 	// 27f, 28h
	//float v = 27.998278f; // 27, 28
	//float v = 27.995278f;  // 27, 28
	//float v = 27.994278f;  //  27, 28
	//float v = 27.994f;  //  27, 28
	//float v = 27.991f;  //  27, 27 
	//float v = 27.992f;  //  27, 27
	//float v = 27.993f;  //  27, 28 
	//float v = 27.9921f;  //  27, 
	float v = 27.9922f;  //  27, 
	//float v = 27.99f;  //  27, 27 
    x[0] = floorf(v);

	half vHalf = __float2half(v);
	float vHalf_fp32 = __half2float(vHalf);

	printf("fp32: %f, fp16 %f, diff %f\n", v, vHalf_fp32, vHalf_fp32 - v);

    y[0] = hfloor(__float2half(v));
}

int main(int argc, char** argv) {

  int devid =0;

  if (argc != 2) {
      fprintf(stderr, "Specify device to use only. (./program devid)\n");
      exit(1);
  }else {
  	devid = atoi(argv[1]);
  }

  cudaSetDevice(devid);

  cudaDeviceProp prop;                                                    
  cudaGetDeviceProperties(&prop, devid);                                 
  printf("device %d : %s\n", devid, prop.name);

  float *x;
  checkCuda(cudaMallocManaged(&x, sizeof(float)));

  half *y;
  checkCuda(cudaMallocManaged(&y, sizeof(half)));
  
  myTest<<<1, 1>>>(x, y);

  checkCuda(cudaDeviceSynchronize());
  
  // check results
  printf("%f\n", x[0]);
  printf("%f\n", half_to_float(y[0]));



  return 0;
}
