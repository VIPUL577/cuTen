#include <cuda.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <vector>

// git ls-files | xargs wc -l to count the number of lines


// They are not as efficient as i would like them to be. but still it works, will improve in near future. hojayega
namespace seera_cuda
{

  __global__ void Reductionsum(float *arr, float *output, int limit, int stride,
                               float divisor, int totalthreads)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalthreads) return;

    float temp = 0.0f;
    int inner = tid % stride;
    int outer = tid / stride;
    int base = outer * limit + inner;

    for (int i = 0; i < limit; i += stride)
    {
      temp = temp + arr[base + i];
    }

    output[tid] = temp / divisor;
  }

  __global__ void Reductionmax(float *arr, float *output, int limit, int stride,
                               float dummy, int totalthreads)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalthreads) return;

    int inner = tid % stride;
    int outer = tid / stride;
    int base = outer * limit + inner;
    float temp = arr[base];

    for (int i = 0; i < limit; i += stride)
    {
      temp = fmaxf(temp, arr[base + i]);
    }

    output[tid] = temp;
  }
  __global__ void Reductionmin(float *arr, float *output, int limit, int stride,
                               float dummy, int totalthreads)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalthreads) return;

    int inner = tid % stride;
    int outer = tid / stride;
    int base = outer * limit + inner;
    float temp = arr[base];

    for (int i = 0; i < limit; i += stride)
    {
      temp = fminf(temp, arr[base + i]);
    }

    output[tid] = temp;
  }
  __global__ void Reductionargmin(float *arr, int *output, int limit, int stride,
                                  int totalthreads)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalthreads) return;

    int arg = 0;
    int inner = tid % stride;
    int outer = tid / stride;
    int base = outer * limit + inner;
    float temp = arr[base];

    for (int i = 0; i < limit; i += stride)
    {
      if (arr[base + i] <= temp)
      {
        temp = arr[base + i];
        arg = i / stride;
      }
    }

    output[tid] = arg;
  }
  __global__ void Reductionargmax(float *arr, int *output, int limit, int stride,
                                  int totalthreads)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalthreads) return;

    int arg = 0;
    int inner = tid % stride;
    int outer = tid / stride;
    int base = outer * limit + inner;
    float temp = arr[base];

    for (int i = 0; i < limit; i += stride)
    {
      if (arr[base + i] >= temp)
      {
        temp = arr[base + i];
        arg = i / stride;
      }
    }

    output[tid] = arg;
  }

  template <typename Kernel>
  void _cuda_reduce_gputogpu(float *A, float *out, int ndims, int dim, int *dimarr,
                             float divisor, Kernel kernel)
  {
    int prod = 1;
    int stride = 1;
    int limit = 1;
    int totalthreads = 1;

    for (int i = ndims - 1; i >= 0; i--)
    {
      prod *= dimarr[i];
      if (i == dim)
        limit = prod;
      else
        totalthreads *= dimarr[i];
    }

    stride = limit / dimarr[dim];

    // out must already be allocated: size = totalthreads

    int threadsPerBlock = 256;
    int blocks = (totalthreads + threadsPerBlock - 1) / threadsPerBlock;

    kernel<<<blocks, threadsPerBlock>>>(A, out, limit, stride, divisor, totalthreads);

    cudaDeviceSynchronize();
  }

  template <typename Kernel>
  void _cuda_reduce_arg_gputogpu(float *A, int *out, int ndims, int dim,
                                 int *dimarr, Kernel kernel)
  {
    int prod = 1, stride = 1, limit = 1, totalthreads = 1;

    for (int i = ndims - 1; i >= 0; i--)
    {
      prod *= dimarr[i];
      if (i == dim)
        limit = prod;
      else
        totalthreads *= dimarr[i];
    }

    stride = limit / dimarr[dim];

    // out must already be allocated: size = totalthreads

    int threads = 256;
    int blocks = (totalthreads + threads - 1) / threads;

    kernel<<<blocks, threads>>>(A, out, limit, stride, totalthreads);

    cudaDeviceSynchronize();
  }

  void cuda_sum  (float *A, float *out, int ndims, int dim, int *dimarr)
  {
    _cuda_reduce_gputogpu(A, out, ndims, dim, dimarr, 1.0f,
                          Reductionsum);
  }

  void cuda_mean  (float *A, float *out, int ndims, int dim, int *dimarr)
  {
    _cuda_reduce_gputogpu(A, out, ndims, dim, dimarr,
                          (float)dimarr[dim], Reductionsum);
  }

  void cuda_max  (float *A, float *out, int ndims, int dim, int *dimarr)
  {
    _cuda_reduce_gputogpu(A, out, ndims, dim, dimarr, 0.0f,
                          Reductionmax);
  }

  void cuda_min  (float *A, float *out, int ndims, int dim, int *dimarr)
  {
    _cuda_reduce_gputogpu(A, out, ndims, dim, dimarr, 0.0f,
                          Reductionmin);
  }

  void cuda_argmax  (float *A, int *out, int ndims, int dim, int *dimarr)
  {
    _cuda_reduce_arg_gputogpu(A, out, ndims, dim, dimarr, Reductionargmax);
  }

  void cuda_argmin  (float *A, int *out, int ndims, int dim, int *dimarr)
  {
    _cuda_reduce_arg_gputogpu(A, out, ndims, dim, dimarr, Reductionargmin);
  }

} // namespace seera_cuda