<div align="center">

# ⚡ cuTen

**A GPU-native tensor library written from scratch in CUDA C++ & Python**

Hand-crafted CUDA kernels · Tensor Core WMMA acceleration · Zero-dependency GPU compute

[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)
[![LOC](https://img.shields.io/badge/CUDA_+_C++-~3400_LOC-orange?style=for-the-badge)]()

</div>

---

**cuTen** is a lightweight, from-scratch GPU tensor library that bypasses cuBLAS/cuDNN entirely. Every kernel — from GEMM to Conv2D to reductions — is hand-written in CUDA C++, many leveraging **NVIDIA Tensor Core WMMA intrinsics** for hardware-accelerated mixed-precision matrix math (FP16 compute → FP32 accumulate). The entire engine is exposed to Python through pybind11 as a single importable module.

> **This is not a wrapper.** There are no calls to cuBLAS, cuDNN, or any external compute library.
> Every operation hits a custom kernel written from the ground up.

---

## ✦ Key Highlights

| | |
|---|---|
| 🧠 **Tensor Core WMMA** | GEMM, Conv2D, and ConvTranspose2D use `nvcuda::wmma` 16×16×16 fragments for hardware-level matrix acceleration |
| 🔥 **Pure CUDA kernels** | ~3400 lines of hand-written `.cu` — no cuBLAS, no cuDNN, no shortcuts |
| 🐍 **Pythonic API** | NumPy-like tensor class with operator overloading (`+`, `*`, `@`, `/`, `**`) |
| 📐 **Full broadcasting** | 4D broadcast add & multiply with automatic shape resolution |
| ⚡ **Fused im2col** | Convolution avoids explicit im2col materialization — im2col is fused directly into the WMMA tiling loop |
| 🧹 **RAII memory** | Automatic GPU memory management via `__del__` destructor with ownership tracking |

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Python (cuTen.py)               │
│   cuten class  ·  operator overloads  ·  .T      │
│   .conv2d()  .matmul()  .relu()  .softmax()     │
└────────────────────┬────────────────────────────┘
                     │  pybind11
┌────────────────────▼────────────────────────────┐
│             seera_cuda (C++ bindings)            │
│         cuda_bindings.cpp  ·  30 KB              │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│              CUDA Kernel Engine                  │
│                                                  │
│  GEMM.cu          ← Tensor Core WMMA matmul      │
│  convolution.cu   ← Fused im2col + WMMA conv2d   │
│  upsampling.cu    ← ConvTranspose2D (WMMA)       │
│  activations.cu   ← ReLU, Sigmoid, Tanh, etc.   │
│  reductionKernels.cu ← Sum, Mean, Max, Min       │
│  maxPool.cu       ← MaxPool2D with mask          │
│  unpooling.cu     ← Nearest-neighbour upsample   │
│  broadcast.cu     ← 4D broadcast add/mul         │
│  elemops.cu       ← Element-wise +, -, *, /      │
│  cuTen_essentails.cu ← Scalar ops, fill, zeros   │
│  col2im.cu        ← col2im reconstruction        │
└──────────────────────────────────────────────────┘
```

---

## Requirements

| Dependency | Version |
|---|---|
| **NVIDIA GPU** | Compute Capability ≥ 7.0 (Tensor Cores required — Volta, Turing, Ampere, Ada, Hopper) |
| **CUDA Toolkit** | 12.x (nvcc must be on `PATH`) |
| **Python** | ≥ 3.10 |
| **pybind11** | `pip install pybind11` |
| **NumPy** | `pip install numpy` |

---

## Installation

### 1. Clone

```bash
git clone https://github.com/<your-username>/cuTen.git
cd cuTen
```

### 2. Install Python dependencies

```bash
pip install pybind11 numpy
```

### 3. Build the CUDA engine

```bash
python build_engine_cuda.py
```

The build script will:

1. Compile each `.cu` kernel into an object file with `nvcc -O3 -arch=sm_89`
2. Compile the pybind11 bindings (`cuda_bindings.cpp`)
3. Link everything into `seera_cuda.cpython-*.so`
4. Run a quick import test to verify the build

> **Note:** Change `-arch=sm_89` in `build_engine_cuda.py` to match your GPU's compute capability (e.g., `sm_70` for V100, `sm_80` for A100, `sm_86` for RTX 3090).

### 4. Verify

```python
from cuTen import cuten
import numpy as np

a = cuten(np.random.randn(2, 3).astype('float32'))
b = cuten(np.random.randn(3, 4).astype('float32'))
c = a @ b  # Tensor Core WMMA matmul on GPU
print(c)
```

---

## Usage Guide

### Creating Tensors

```python
from cuTen import cuten
import numpy as np

# From NumPy array  →  auto-transfers to GPU
x = cuten(np.random.randn(4, 3, 28, 28).astype('float32'))

# From Python list
v = cuten([1.0, 2.0, 3.0])

# Factory methods (allocated directly on GPU, no host round-trip)
z = cuten.zeros((64, 128))
o = cuten.ones((32, 10))
```

### Arithmetic (Operator Overloading)

All operators return **new** tensors — cuTen never mutates in-place.

```python
a = cuten(np.random.randn(4, 4).astype('float32'))
b = cuten(np.random.randn(4, 4).astype('float32'))

c = a + b          # element-wise add
d = a * b          # element-wise multiply
e = a - b          # subtract
f = a / b          # divide
g = a ** 2         # power
h = a + 3.14       # scalar add
i = a * 0.01       # scalar multiply
```

### Broadcasting

Supports NumPy-style 4D broadcasting rules:

```python
x = cuten(np.random.randn(8, 64, 14, 14).astype('float32'))   # (N, C, H, W)
bias = cuten(np.random.randn(1, 64, 1, 1).astype('float32'))  # (1, C, 1, 1)

y = x + bias   # broadcast add across batch & spatial dims
```

### Matrix Multiplication (Tensor Core WMMA)

```python
# 2D matmul
A = cuten(np.random.randn(128, 64).astype('float32'))
B = cuten(np.random.randn(64, 256).astype('float32'))
C = A @ B      # (128, 256) — runs on Tensor Cores

# Batched matmul
A = cuten(np.random.randn(8, 128, 64).astype('float32'))
B = cuten(np.random.randn(64, 256).astype('float32'))
C = A @ B      # (8, 128, 256) — batched WMMA
```

### Transpose

```python
x = cuten(np.random.randn(128, 64).astype('float32'))
y = x.T        # (64, 128)  — GPU-side transpose

# 3D batched transpose
x = cuten(np.random.randn(8, 128, 64).astype('float32'))
y = x.T        # (8, 64, 128) — transposes last two dims per batch
```

### Activations

```python
x = cuten(np.random.randn(32, 128).astype('float32'))

y = x.relu()
y = x.sigmoid()
y = x.tanh()
y = x.exp()
y = x.log()
y = x.abs()
y = x.sqrt()
y = x.clip(-1.0, 1.0)
```

### Softmax

```python
logits = cuten(np.random.randn(32, 10).astype('float32'))
probs = logits.softmax()   # softmax over last dimension
```

### Reductions

```python
x = cuten(np.random.randn(8, 64, 14, 14).astype('float32'))

s = x.sum(dim=1)    # sum along channels   → (8, 14, 14)
m = x.mean(dim=0)   # mean along batch     → (64, 14, 14)
mx = x.max(dim=2)   # max along height     → (8, 64, 14)
mn = x.min(dim=3)   # min along width      → (8, 64, 14)
```

### Conv2D (Fused im2col + WMMA)

```python
inp = cuten(np.random.randn(2, 3, 28, 28).astype('float32'))     # (N, C, H, W)
kernel = cuten(np.random.randn(16, 3, 5, 5).astype('float32'))   # (F, C, KH, KW)

out = inp.conv2d(kernel, strideh=1, stridew=1, padh=2, padw=2)   # (2, 16, 28, 28)
```

### ConvTranspose2D

```python
inp = cuten(np.random.randn(2, 16, 14, 14).astype('float32'))    # (N, Cin, H, W)
kernel = cuten(np.random.randn(16, 32, 3, 3).astype('float32'))  # (Cin, Cout, KH, KW)

out = inp.conv2d_transpose(kernel, strideh=2, stridew=2, padh=1, padw=1)
# → (2, 32, 27, 27)
```

### MaxPool2D

Returns both the pooled output and the argmax mask (stored on GPU as `int32`):

```python
inp = cuten(np.random.randn(2, 16, 28, 28).astype('float32'))

pooled, mask = inp.maxpool2d(KH=2, KW=2, strideh=2, stridew=2)
# pooled: (2, 16, 14, 14)
# mask:   (2, 16, 28, 28)  — binary mask for backward pass
```

### Nearest-Neighbour Unpooling

```python
small = cuten(np.random.randn(2, 16, 7, 7).astype('float32'))

big = small.unpool(sh=2, sw=2)   # (2, 16, 14, 14)
```

### Concatenation

```python
a = cuten(np.random.randn(4, 32, 14, 14).astype('float32'))
b = cuten(np.random.randn(4, 16, 14, 14).astype('float32'))

c = a.concatenate2D(b)   # (4, 48, 14, 14) — concat along channels

# 1D concatenation
x = cuten(np.random.randn(4, 100).astype('float32'))
y = cuten(np.random.randn(4, 50).astype('float32'))
z = x.concatenate1D(y)   # (4, 150)
```

### Reshape & Flatten

```python
x = cuten(np.random.randn(8, 16, 7, 7).astype('float32'))

flat = x.flatten()                     # (8, 784) — preserves batch dim
reshaped = x.reshape((8, 16 * 7 * 7)) # same as flatten for this shape
```

### Host Transfer

```python
x = cuten(np.random.randn(4, 4).astype('float32'))

np_arr = x.to_host_f32()   # returns np.ndarray on CPU
```

---

## Supported Dtypes

| dtype | GPU Alloc | Host Transfer |
|---|---|---|
| `float32` | ✅ | `to_host_f32()` |
| `int32` | ✅ | `to_host_i32()` |
| `int16` | ✅ (alloc only) | — |

---

## Project Structure

```
cuTen/
├── cuTen.py                  # Python tensor class (cuten)
├── build_engine_cuda.py      # Build system — compiles & links all CUDA code
├── engine_cuda/
│   ├── include/
│   │   └── seera_engine_cuda.hpp   # C++ header (all kernel declarations)
│   └── src/
│       ├── GEMM.cu                 # Batched matmul — WMMA Tensor Cores
│       ├── convolution.cu          # Conv2D fwd/bwd — fused im2col + WMMA
│       ├── upsampling.cu           # ConvTranspose2D fwd/bwd — WMMA
│       ├── activations.cu          # ReLU, Sigmoid, Tanh, Log, Exp, Abs, Sqrt, Clip, Softmax
│       ├── reductionKernels.cu     # Sum, Mean, Max, Min, ArgMax, ArgMin (any-dim reduction)
│       ├── maxPool.cu              # MaxPool2D fwd/bwd with argmax mask
│       ├── unpooling.cu            # Nearest-neighbour upsample fwd/bwd
│       ├── elemops.cu              # Element-wise add, sub, mul, div
│       ├── broadcast.cu            # 4D broadcast add & multiply
│       ├── cuTen_essentails.cu     # Scalar ops, fill, zeros, ones, power
│       ├── col2im.cu               # col2im fold for transpose conv
│       └── cuda_bindings.cpp       # pybind11 bindings — bridges CUDA ↔ Python
└── seera_cuda.cpython-*.so         # compiled shared library (after build)
```

---

## How the WMMA Kernels Work

The performance-critical kernels (GEMM, Conv2D, ConvTranspose2D) use NVIDIA's **Warp Matrix Multiply-Accumulate** API:

```
1.  Load 16×16 tiles from global memory → shared memory (FP32 → FP16 conversion)
2.  wmma::load_matrix_sync() — load tiles into WMMA fragments
3.  wmma::mma_sync()         — hardware 16×16×16 matmul (FP16 × FP16 + FP32)
4.  Slide the K-dimension window and accumulate
5.  wmma::store_matrix_sync() — write FP32 result back to shared memory
6.  Cooperative store to global memory with bounds checking
```

The convolution kernel **fuses im2col into the tiling loop** — it never materializes the full im2col matrix, computing patch indices on-the-fly within the shared memory load step. This saves significant memory bandwidth.

---

## License

MIT

---

<div align="center">
<sub>Built with raw CUDA, caffeine, and an unhealthy obsession with Tensor Cores.</sub>
</div>
