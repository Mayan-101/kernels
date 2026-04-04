# CUDA Kernels from Scratch includes SGeMM Kernels , Softmaxx kernels and a naive MNIST NN implementation

Hand-written CUDA kernels for matrix multiplication, softmax, and attention — built progressively from naive implementations to highly optimized tiled/vectorized variants, with an auto-tuner and a full MNIST neural network.



---

## Kernels

### Matrix Multiplication (`matmul_bench.cu`)

Four progressively optimized kernels, benchmarked head-to-head against cuBLAS:

| Kernel | Strategy |
|---|---|
| `matmul` | Naive 2D thread per output element |
| `matmul_1D` | Coalesced 1D thread indexing |
| `matmul_tiled` | Shared memory tiling |
| `matmul_tiled_1D_coarse` | 1D tiling + thread coarsening (TM output rows per thread) |
| `matmul_tiled_2D_coarse_vec` | 2D tiling + register blocking (TM×TN) + `float4` vectorized loads, transposed A in SMEM |

The final kernel (`tiled_2D_coarse_vec`) stores A transposed in shared memory to enable conflict-free column access during the dot product loop and uses `float4` loads for both A and B to maximize memory bandwidth.
### Matrix Multiplication (square, N×N)

> Tested on NVIDIA GeForce MX330 (Pascal Architecture) GPU. All times in ms, bandwidth in GB/s.


### Softmax (`softmax_bench.cu`, `softmax_playground.cu`)

| Kernel | Strategy |
|---|---|
| `softmax_naive` | One thread per row, single-pass with global memory |
| `softmax_shared` | Online (numerically stable) softmax with shared memory tiling — single-pass max+denominator update |

The shared kernel implements the **online softmax** algorithm: it accumulates the running max and denominator in a single pass over tiles, rescaling the denominator whenever the max increases. This avoids a second pass over the data.


### MNIST Neural Network (`nn.cu`)

A complete 3-layer MLP trained on MNIST written entirely in CUDA — no PyTorch, no cuDNN.

**Architecture:** `784 → 256 → 128 → 10`

Kernels implemented from scratch:
- `FeedforwardReLU` / `Feedforward` — forward pass with and without activation
- `softmax_shared` — numerically stable online softmax
- `crossEntropyLoss` — per-sample cross-entropy
- `init_weights` — He initialization via `cuRAND`
- `feedforwarrelu_backward` / `feedforwarlinear_backward` — gradient w.r.t. weights & biases
- `feedforwarrelu_backwardx` / `feedforwarlinear_backwardx` — gradient w.r.t. input activations
- `crossEntropyBackwards` — fused softmax + cross-entropy gradient
- `update_layer` — SGD weight update

---

## Benchmarks



### Softmax (online shared-memory vs naive)

> All kernels verified to pass numerical correctness checks (max error < 1e-5).

```
shape (RxC)            naive (ms)    shared (ms)
────────────────────────────────────────────────
4096 x 4096             67.917          0.014
2048 x 8192             28.496          0.014
512  x 1024              1.094          0.005
4096 x 16192           774.918          0.005
```

The shared-memory online kernel is **~4800–155000× faster** than the naive implementation across shapes, driven by dramatically reduced global memory traffic.

---

## Auto-Tuner (`tune.sh`)

Sweeps the tiling parameter space for `matmul_tiled_2D_coarse_vec` and finds the fastest valid configuration.

**Parameters swept:**

| Parameter | Values | Role |
|---|---|---|
| `BM`, `BN` | 32, 64, 128 | Output tile per CTA |
| `BK` | 8, 16, 32 | Phase depth along K |
| `TM`, `TN` | 4, 8 | Per-thread register tile |
| `BLOCK_SIZE` | 32, 64, 128, 256, 512 | Loading granularity (controls stride_A/stride_B) |

Validity constraints checked before compilation (compute thread count, float4 alignment, stride divisibility, 48 KB shared memory limit). Each configuration is compiled with `nvcc`, run with a timeout, and timed.

```bash
./tune.sh
```

---

## Building

```bash
# Matrix multiplication benchmark
nvcc -O3 -o matmul_bench matmul_bench.cu -lcublas && ./matmul_bench

# Softmax benchmark
nvcc -O3 -o softmax_bench softmax_bench.cu && ./softmax_bench

# MNIST neural network (requires data/mnist_train.csv and data/mnist_test.csv)
nvcc -O3 -o nn nn.cu -lcurand && ./nn

# Auto-tuner (targets finetune_kernel.cu)
chmod +x tune.sh && ./tune.sh
```

---

## MNIST Data

Place CSV files in a `data/` directory:

```
data/
├── mnist_train.csv   # 60 000 rows, format: label,px0,...,px783
└── mnist_test.csv    # 10 000 rows, same format
```

Available from [Kaggle MNIST in CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv).
