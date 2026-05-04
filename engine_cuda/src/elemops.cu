#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#include <stdlib.h>
namespace seera_cuda
{

    // =======================
    // Kernels
    // =======================

    __global__ void elemadd(float *A, float *B, float *C, int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
            C[idx] = A[idx] + B[idx];
    }

    __global__ void elemsub(float *A, float *B, float *C, int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
            C[idx] = A[idx] - B[idx];
    }

    __global__ void elemdiv(float *A, float *B, float *C, int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
            C[idx] = A[idx] / B[idx];
    }

    __global__ void elemmult(float *A, float *B, float *C, int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
            C[idx] = A[idx] * B[idx];
    }


    void cuda_elemadd_gputogpu(float *A, float *B, float *C, int size)
    {
        int threads = 256;
        int blocks = (size + threads - 1) / threads;

        elemadd<<<blocks, threads>>>(A, B, C, size);
        cudaDeviceSynchronize();
    }

    void cuda_elemsub_gputogpu(float *A, float *B, float *C, int size)
    {
        int threads = 256;
        int blocks = (size + threads - 1) / threads;

        elemsub<<<blocks, threads>>>(A, B, C, size);
        cudaDeviceSynchronize();
    }

    void cuda_elemdiv_gputogpu(float *A, float *B, float *C, int size)
    {
        int threads = 256;
        int blocks = (size + threads - 1) / threads;

        elemdiv<<<blocks, threads>>>(A, B, C, size);
        cudaDeviceSynchronize();
    }

    void cuda_elemmult_gputogpu(float *A, float *B, float *C, int size)
    {
        int threads = 256;
        int blocks = (size + threads - 1) / threads;

        elemmult<<<blocks, threads>>>(A, B, C, size);
        cudaDeviceSynchronize();
    }


} // namespace seera_cuda