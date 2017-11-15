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

__device__ inline half h_fabs(half v) {
    return (__hgeu(v,__float2half(0.0f))?v:__hneg(v));
}

__global__
void myTest(float *x, half *y)
{
    y[0] = h_fabs(__float2half(x[0]));
}

__global__
void myTest_v1(float *x, float *y)
{
    y[0] = __half2float(h_fabs(__float2half(x[0])));
}



void mytest(float v)
{
  float *x;
  checkCuda(cudaMallocManaged(&x, sizeof(float)));

  half *y;
  checkCuda(cudaMallocManaged(&y, sizeof(half)));

  x[0] = v;
  printf("input\t : %f\n", x[0]);
  
  myTest<<<1, 1>>>(x, y);

  checkCuda(cudaDeviceSynchronize());
  
  // check results
  printf("output\t : %f\n", half_to_float(y[0]));

  cudaFree(x);
  cudaFree(y);
}


void mytest_v1(float v)
{
  float *x;
  checkCuda(cudaMallocManaged(&x, sizeof(float)));

  float *y;
  checkCuda(cudaMallocManaged(&y, sizeof(float)));

  x[0] = v;
  printf("input\t : %f\n", x[0]);

  half tmpV = approx_float_to_half(x[0]);
  printf("input (approx half)\t : %f\n", tmpV );
  printf("input (approx half, back to float)\t : %f\n", half_to_float(tmpV) );
  
  myTest_v1<<<1, 1>>>(x, y);

  checkCuda(cudaDeviceSynchronize());
  
  // check results
  printf("output(native __half2float)\t : %f\n", y[0]);

  cudaFree(x);
  cudaFree(y);
}

__global__ void Kern_half2_abs(float *x)
{
	half2 v2 = __floats2half2_rn(1.1f, -2.1f);

	printf("value in half2 : %f %f\n", __half2float(__low2half(v2)),
			__half2float(__high2half(v2)));

	// abs()
	*((int*)(&v2)) &= 0x7FFF7FFF;

	printf("value in half2 (abs) : %f %f\n", __half2float(__low2half(v2)),
			__half2float(__high2half(v2)));

}


void half2_abs(void)
{
  float *x;
  checkCuda(cudaMallocManaged(&x, sizeof(float)));

  Kern_half2_abs<<<1, 1>>>(x);


  checkCuda(cudaDeviceSynchronize());
  cudaFree(x);
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


  /*
  printf("\ntest 1\n");
  float v_pos = 1.111f;
  float v_neg = -1.111f;

  mytest(v_pos);
  mytest(v_neg);
  mytest_v1(v_pos);
  mytest_v1(v_neg);

  printf("\ntest 2\n");
  v_pos = 1.11f;
  v_neg = -1.11f;

  mytest(v_pos);
  mytest(v_neg);
  mytest_v1(v_pos);
  mytest_v1(v_neg);

  printf("\ntest 3\n");
  v_pos = 1.3333f;
  v_neg = -1.3333f;

  mytest(v_pos);
  mytest(v_neg);
  mytest_v1(v_pos);
  mytest_v1(v_neg);
  */


  half2_abs();

  return 0;
}
