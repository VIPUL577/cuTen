#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
namespace seera_cuda
{
    __global__ void unpooling(float *inp, float *out, int N, int C, int H, int W, int sh, int sw) //-> launch with H*W(incoming)
    {
        int H_out = sh * H;
        int W_out = sw * W;

        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int w_out = index % W_out;
        int h_out = index / W_out;
        int c = blockIdx.y;
        int batchN = blockIdx.z;

        if (h_out >= 0 && h_out < H_out && w_out >= 0 && w_out < W_out)
        {
            int h_in = h_out / sh;
            int w_in = w_out / sw;

            int common = batchN * C + c;
            out[common * H_out * W_out + h_out * W_out + w_out] = inp[common * H * W + h_in * W + w_in];
        }
    }



    void cuda_unpooling(float *d_inp, float *d_out,
                                 int batchN, int C, int H, int W,
                                 int sh, int sw)
    {
        int H_out = sh * H;
        int W_out = sw * W;

        int output_elems = batchN * C * H_out * W_out;
        int spatial_out = H_out * W_out;

        // assume d_out already allocated
        cudaMemset(d_out, 0, sizeof(float) * output_elems);

        int threads = 256;
        int Nblock = (spatial_out + threads - 1) / threads;
        dim3 blocks(Nblock, C, batchN);

        unpooling<<<blocks, threads>>>(d_inp, d_out,
                                       batchN, C, H, W,
                                       sh, sw);

        cudaDeviceSynchronize();
    }

}