#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
namespace seera_cuda {
using namespace nvcuda;

// ============================================================================
// FORWARD PASS: Conv2dTranspose
// ============================================================================
// X(Batch,Cin,Hin,Win) * W(Cin,Cout,R,S) -> (Batch,Cout,Hout,Wout)
__global__ void
conv2dTransmatmul(float *X, float *W, float *C, int M, int N, int K, int Cout,
                  int R, int S, int BATCH, int Hin,
                  int Win) { // k-> Cin, M-> Cout*R*S, N-> Batch*H_in*W_in
  int warpM = blockIdx.y * 16;
  int warpN = blockIdx.x * 16;

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

  wmma::fill_fragment(acc_frag, 0.0f);

  __shared__ half shA[16 * 16];
  __shared__ half shB[16 * 16];

  for (int p = 0; p < K; p += 16) {
    for (int i = 0; i < 8; ++i) {
      int linear_idx = threadIdx.x + i * 32;
      int row = linear_idx / 16;
      int col = linear_idx % 16;

      int global_row_A = warpM + row;
      int A_s = global_row_A % S;
      int A_r = (global_row_A / S) % R;
      int A_cout = global_row_A / (S * R);

      int A_cin = p + col;
      if (global_row_A < M && A_cin < K) {
        shA[linear_idx] = __float2half(
            W[A_cin * (Cout * R * S) + A_cout * (R * S) + A_r * S + A_s]);
      } else {
        shA[linear_idx] = __float2half(0.0f);
      }

      int B_cin = p + row;
      int global_col_B = warpN + col;
      int B_w = global_col_B % Win;
      int B_h = (global_col_B / Win) % Hin;
      int B_b = global_col_B / (Win * Hin);
      if (B_cin < K && global_col_B < N) {
        shB[linear_idx] = __float2half(
            X[B_b * K * Hin * Win + B_cin * Hin * Win + B_h * Win + B_w]);
      } else {
        shB[linear_idx] = __float2half(0.0f);
      }
    }

    __syncthreads();

    wmma::load_matrix_sync(a_frag, shA, 16);
    wmma::load_matrix_sync(b_frag, shB, 16);

    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    __syncthreads();
  }

  __shared__ float shC[16 * 16];
  wmma::store_matrix_sync(shC, acc_frag, 16, wmma::mem_row_major);
  __syncthreads();

  for (int i = 0; i < 8; ++i) {
    int linear_idx = threadIdx.x + i * 32;
    int row = linear_idx / 16;
    int col = linear_idx % 16;

    int global_row_C = warpM + row;
    int global_col_C = warpN + col;

    if (global_row_C < M && global_col_C < N) {
      C[(global_row_C)*N + global_col_C] = shC[linear_idx];
    }
  }
}

__global__ void col2im(float *d_out, float *d_in, int C, int R, int S, int H_in,
                       int W_in, int H_out, int W_out, int stridew, int strideh,
                       int padh, int padw, int total_N) {
  int globalid = blockIdx.x * blockDim.x + threadIdx.x;
  int c = blockIdx.y;
  int batchN = blockIdx.z;
  int sr = S * R;
  int w_in = globalid % W_in;
  int h_in = globalid / W_in;

  if (h_in >= H_in || w_in >= W_in || c >= C)
    return;

  float temp = 0.0f;
  for (int row = 0; row < sr; row++) {
    int s = row % S;
    int r = row / S;

    int h_num = h_in + padh - r;
    int w_num = w_in + padw - s;

    if (h_num % strideh != 0 || w_num % stridew != 0)
      continue;

    int h_out = h_num / strideh;
    int w_out = w_num / stridew;

    if (h_out >= 0 && h_out < H_out && w_out >= 0 && w_out < W_out) {
      int gemm_row = c * sr + row;
      int gemm_col = batchN * H_out * W_out + h_out * W_out + w_out;
      temp += d_in[gemm_row * total_N + gemm_col];
    }
  }

  d_out[(batchN * C + c) * H_in * W_in + globalid] = temp;
}



void cuda_col2im_gputogpu(float *d_in, float *d_out, int batchN, int C,
                          int H_in, int W_in, int R, int S, int pad_h,
                          int pad_w, int stride_h, int stride_w, int total_N) {
  int H_out = (H_in + 2 * pad_h - R) / stride_h + 1;
  int W_out = (W_in + 2 * pad_w - S) / stride_w + 1;

  int tpb = 256;

  dim3 grid((H_in * W_in + tpb - 1) / tpb, C, batchN);
  dim3 block(tpb);

  col2im<<<grid, block>>>(d_out, d_in, C, R, S, H_in, W_in, H_out, W_out,
                          stride_w, stride_h, pad_h, pad_w, total_N);

  cudaDeviceSynchronize();
}


void cuda_conv2DTranpose(float *hA, float *hB, float *hC, int batch,
                             int Cin, int Hin, int Win, int Cout, int KH,
                             int KW, int strideh, int stridew, int padh,
                             int padw) {
  int Hout = (Hin - 1) * strideh - 2 * padh + KH;
  int Wout = (Win - 1) * stridew - 2 * padw + KW;

  int N = batch * Hin * Win;
  int M = Cout * KH * KW;
  int K = Cin;

  float *intermediate;
  cudaMalloc(&intermediate, M * N * sizeof(float));

  dim3 block(32);
  dim3 grid((N + 15) / 16, (M + 15) / 16);

  conv2dTransmatmul<<<grid, block>>>(hA, hB, intermediate, M, N, K, Cout, KH,
                                     KW, batch, Hin, Win);
  cudaDeviceSynchronize();

  cuda_col2im_gputogpu(intermediate, hC, batch, Cout, Hout, Wout, KH, KW, padh,
                       padw, strideh, stridew, N);

  cudaFree(intermediate);
}

} // namespace seera_cuda