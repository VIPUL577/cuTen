

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <mma.h>
#include <time.h>
#include <vector>
namespace seera_cuda
{
    using namespace nvcuda;

    __global__ void convulution_eff(const float *input_image, float *conv,
                                    float *kernel, int N, int C, int H, int W,
                                    int R, int S, int pad_h, int pad_w,
                                    int stride_h, int stride_w, int H_out,
                                    int W_out)
    {
        __shared__ half iw2col[16 * 16];
        __shared__ half krl[16 * 16];

        int batchno = blockIdx.z;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> im2col_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> krl_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

        wmma::fill_fragment(acc_frag, 0.0f);
        for (int p = 0; p < C * S * R; p += 16)
        {
            for (int i = 0; i < 8; i++)
            {
                int tid = threadIdx.x + i * 32; //-> 255 - unique

                int local_row = (tid / 16); // for row divide hota hai
                int local_col = (tid % 16); // for column mod hota hai

                int global_image = blockIdx.x * 16 + local_row; // -> H_out*W_out
                int ni = blockIdx.y * 16 + local_col;           // -> N
                int global_kernel_ = p + local_row;             // -> C*S*R
                int global_kernel = p + local_col;

                int iy_w_out = (global_image % W_out);
                int iy_h_out = global_image / W_out;

                int ky_is = global_kernel % S;
                int index_ = global_kernel / S;
                int ky_ir = index_ % R;
                int ky_ic = (index_ / R);

                int h_in = iy_h_out * stride_h - pad_h + ky_ir;
                int w_in = iy_w_out * stride_w - pad_w + ky_is;

                if (ni < N && ky_is < S && ky_ir < R)
                {
                    int kernel_idx = (ni) * (C * S * R) +
                                     global_kernel_; // isiliye since kernel ka row is used
                    krl[tid] = __float2half(kernel[kernel_idx]);
                }
                else
                {
                    krl[tid] = __float2half(0.0f);
                }
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W && ky_ic < C)
                {
                    int input_idx = ((batchno * C + ky_ic) * H + h_in) * W + w_in;
                    iw2col[tid] = __float2half(
                        input_image[input_idx]); //- half(1); // and subsequntly im2col
                                                 // matrix ka column use ho raha hai.
                }
                else
                {
                    iw2col[tid] = __float2half(0.0f);
                }
            }

            __syncthreads();

            wmma::load_matrix_sync(im2col_frag, iw2col, 16);
            wmma::load_matrix_sync(krl_frag, krl, 16);

            wmma::mma_sync(acc_frag, im2col_frag, krl_frag, acc_frag);

            __syncthreads();
        }
        __shared__ float sha_conv[16 * 16];
        wmma::store_matrix_sync(sha_conv, acc_frag, 16, wmma::mem_row_major);

        for (int i = 0; i < 8; i++)
        {
            int tid = threadIdx.x + i * 32; //-> 255
            int index_ = blockIdx.x * 16 + (tid / 16);

            int ni = blockIdx.y * 16 + (tid % 16); // -> N

            if (ni < N && index_ < H_out * W_out)
                conv[(batchno * N + ni) * H_out * W_out + index_] =
                    sha_conv[tid];
        }
    }

    __global__ void _weight_reduce(float *dW, float *dwn, int BatchN, int Cin,
                                   int Cout, int R, int S)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x; //-> Cout,Cin,R,S
        int s = index % S;
        int r = (index / S) % R;
        int cout = (index / (S * R)) % Cout;
        int cin = index / (S * R * Cout);
        float temp = 0.0f;
        if (s < S && r < R && cin < Cin && cout < Cout)
        {
            for (int ii = 0; ii < BatchN; ii++)
            {
                temp += dwn[ii * Cout * Cin * R * S + index];
            }
            dW[index] = temp;
        }
    }


  
    void cuda_conv2d(float *h_image, float *h_kernel, float *d_conv, int batchN,
                         int C, int H, int W, int N, int R, int S, int pad_h,
                         int pad_w, int stride_h, int stride_w)
    {
        int H_out = (H + 2 * pad_h - R) / stride_h + 1;
        int W_out = (W + 2 * pad_w - S) / stride_w + 1;

        int input_elems = batchN * C * H * W;
        int kernel_elems = N * C * R * S;
        int output_elems = batchN * N * H_out * W_out;

        // Launch convolution kernel
        int aa1 = ceil((float)H_out * W_out / 16);
        int aa2 = ceil((float)N / 16);
        dim3 tpb(32, 1);
        dim3 block(aa1, aa2, batchN);

        convulution_eff<<<block, tpb>>>(h_image, d_conv, h_kernel, N, C, H, W, R, S,
                                        pad_h, pad_w, stride_h, stride_w, H_out,
                                        W_out);

        cudaDeviceSynchronize();
    }
 

} // namespace seera_cuda