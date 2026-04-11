# 🔬 CUDA Kernel Engine — Benchmarks

Performance comparison of **cuTen's hand-written CUDA kernels** against their **cuBLAS / cuDNN equivalents**.

> All benchmarks run on: **GPU:** `___________` · **CUDA:** `___________` · **Driver:** `___________`
>
> Timing: kernel-only (no host↔device transfer), averaged over 100 warm runs, `cudaDeviceSynchronize()` after each call.

---

## GEMM — Matrix Multiplication

Kernel: `GEMM.cu` → `cuda_matmul()` · **Tensor Core WMMA** (FP16 compute, FP32 accumulate)

| Size (M × K × N) | Batch | cuTen WMMA (ms) | cuBLAS `cublasSgemm` (ms) | cuTen / cuBLAS |
|---|---|---|---|---|
| 128 × 128 × 128 | 1 | | | |
| 256 × 256 × 256 | 1 | | | |
| 512 × 512 × 512 | 1 | | | |
| 1024 × 1024 × 1024 | 1 | | | |
| 2048 × 2048 × 2048 | 1 | | | |
| 4096 × 4096 × 4096 | 1 | | | |
| 128 × 64 × 256 | 8 | | | |
| 128 × 64 × 256 | 32 | | | |

---

## Conv2D Forward

Kernel: `convolution.cu` → `cuda_conv2d_fwd()` · **Fused im2col + WMMA**

| Input (N×C×H×W) | Kernel (F×C×KH×KW) | Stride | Pad | cuTen WMMA (ms) | cuDNN `cudnnConvolutionForward` (ms) | cuTen / cuDNN |
|---|---|---|---|---|---|---|
| 1×3×224×224 | 64×3×3×3 | 1 | 1 | | | |
| 1×3×224×224 | 64×3×7×7 | 2 | 3 | | | |
| 8×3×32×32 | 16×3×3×3 | 1 | 1 | | | |
| 8×3×32×32 | 16×3×5×5 | 1 | 2 | | | |
| 16×64×16×16 | 128×64×3×3 | 1 | 1 | | | |
| 32×64×16×16 | 128×64×3×3 | 1 | 1 | | | |
| 1×128×8×8 | 256×128×3×3 | 1 | 1 | | | |

---

## ConvTranspose2D Forward

Kernel: `upsampling.cu` → `cuda_conv2DTranpose_fwd()` · **WMMA + col2im**

| Input (N×Cin×H×W) | Kernel (Cin×Cout×KH×KW) | Stride | Pad | cuTen WMMA (ms) | cuDNN `cudnnConvolutionBackwardData` (ms) | cuTen / cuDNN |
|---|---|---|---|---|---|---|
| 1×256×8×8 | 256×128×3×3 | 2 | 1 | | | |
| 8×128×16×16 | 128×64×4×4 | 2 | 1 | | | |
| 8×16×14×14 | 16×32×3×3 | 2 | 1 | | | |
| 16×64×7×7 | 64×32×3×3 | 2 | 1 | | | |

---

## MaxPool2D Forward

Kernel: `maxPool.cu` → `cuda_maxpool_fwd()`

| Input (N×C×H×W) | Pool (KH×KW) | Stride | cuTen (ms) | cuDNN `cudnnPoolingForward` (ms) | cuTen / cuDNN |
|---|---|---|---|---|---|
| 1×64×112×112 | 2×2 | 2 | | | |
| 8×64×32×32 | 2×2 | 2 | | | |
| 16×128×16×16 | 2×2 | 2 | | | |
| 32×128×16×16 | 3×3 | 2 | | | |
| 1×64×56×56 | 3×3 | 1 | | | |

---

## Nearest-Neighbour Unpooling (Upsample)

Kernel: `unpooling.cu` → `cuda_unpooling_fwd()`

| Input (N×C×H×W) | Scale (sh×sw) | cuTen (ms) | cuDNN Upsample Nearest (ms) | cuTen / cuDNN |
|---|---|---|---|---|
| 8×64×7×7 | 2×2 | | | |
| 8×128×14×14 | 2×2 | | | |
| 16×256×4×4 | 4×4 | | | |
| 1×64×28×28 | 2×2 | | | |

---

## Activations

Kernel: `activations.cu`

### ReLU

| Size | cuTen (μs) | cuDNN `cudnnActivationForward` (μs) | cuTen / cuDNN |
|---|---|---|---|
| 1,024 | | | |
| 65,536 | | | |
| 1,048,576 | | | |
| 16,777,216 | | | |

### Sigmoid

| Size | cuTen (μs) | cuDNN `cudnnActivationForward` (μs) | cuTen / cuDNN |
|---|---|---|---|
| 1,024 | | | |
| 65,536 | | | |
| 1,048,576 | | | |
| 16,777,216 | | | |

### Tanh

| Size | cuTen (μs) | cuDNN `cudnnActivationForward` (μs) | cuTen / cuDNN |
|---|---|---|---|
| 1,024 | | | |
| 65,536 | | | |
| 1,048,576 | | | |
| 16,777,216 | | | |

---

## Softmax

Kernel: `activations.cu` → `cuda_softmax_fwd()`

| Shape (N × C) | cuTen (μs) | cuDNN `cudnnSoftmaxForward` (μs) | cuTen / cuDNN |
|---|---|---|---|
| 32 × 10 | | | |
| 128 × 10 | | | |
| 128 × 1000 | | | |
| 512 × 1000 | | | |
| 1024 × 10000 | | | |

---

## Element-wise Operations

Kernel: `elemops.cu`

| Operation | Size | cuTen (μs) | cuBLAS / Thrust (μs) | cuTen / Ref |
|---|---|---|---|---|
| Add | 1,048,576 | | | |
| Multiply | 1,048,576 | | | |
| Subtract | 1,048,576 | | | |
| Divide | 1,048,576 | | | |
| Add | 16,777,216 | | | |
| Multiply | 16,777,216 | | | |

---

## Broadcast Operations

Kernel: `broadcast.cu` → `broadcast_add_4d()`, `broadcast_mul_4d()`

| A shape | B shape | Op | cuTen (μs) | PyTorch Equivalent (μs) | cuTen / PyTorch |
|---|---|---|---|---|---|
| (8, 64, 14, 14) | (1, 64, 1, 1) | Add | | | |
| (8, 64, 14, 14) | (1, 64, 1, 1) | Mul | | | |
| (32, 128, 7, 7) | (1, 128, 1, 1) | Add | | | |
| (1, 3, 224, 224) | (1, 3, 1, 1) | Add | | | |

---

## Reductions

Kernel: `reductionKernels.cu`

| Operation | Input Shape | Reduce Dim | cuTen (μs) | cuBLAS / CUB Reference (μs) | cuTen / Ref |
|---|---|---|---|---|---|
| Sum | (8, 64, 14, 14) | 1 | | | |
| Mean | (8, 64, 14, 14) | 0 | | | |
| Max | (8, 64, 14, 14) | 2 | | | |
| Min | (8, 64, 14, 14) | 3 | | | |
| Sum | (32, 1024) | 1 | | | |
| Sum | (1, 16777216) | 0 | | | |

---

## Transpose

Kernel: `GEMM.cu` → `cuda_transpose_2d()`, `cuda_transpose_3d()`

| Shape | cuTen (μs) | cuBLAS `cublasSgeam` (μs) | cuTen / cuBLAS |
|---|---|---|---|
| 1024 × 1024 | | | |
| 2048 × 2048 | | | |
| 4096 × 4096 | | | |
| (8, 512, 256) | | | |
| (32, 128, 64) | | | |

---

## Scalar Operations

Kernel: `cuTen_essentails.cu`

| Operation | Size | cuTen (μs) | Thrust / cuBLAS (μs) | cuTen / Ref |
|---|---|---|---|---|
| Scalar Multiply | 1,048,576 | | | |
| Scalar Add | 1,048,576 | | | |
| Power | 1,048,576 | | | |
| Fill Zeros | 1,048,576 | | | |
| Fill Ones | 1,048,576 | | | |
| Scalar Multiply | 16,777,216 | | | |

---

## Notes

- **WMMA kernels** use FP16 compute → FP32 accumulate (`wmma::fragment<..., half, ...>` + `float` accumulator). This means results may have small numerical differences vs. pure FP32 cuBLAS, but match Tensor Core precision.
- **cuTen Conv2D** fuses im2col into the WMMA tiling loop — it never allocates the full im2col matrix, saving O(C·K²·H_out·W_out) memory.
- The cuDNN reference should use `CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM` or the auto-tuned best algorithm for a fair comparison.
- All cuTen kernels call `cudaDeviceSynchronize()` after each launch. For benchmarking, this ensures timing accuracy but note that in production you'd want to batch launches and minimize syncs.
