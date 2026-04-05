# CUDA Kernels from Scratch includes SGeMM Kernels , Softmax kernels and a naive MNIST NN implementation

Hand-written CUDA kernels for matrix multiplication and softmax— built progressively from naive implementations to highly optimized tiled/vectorized variants, with an auto-tuner and a full MNIST neural network.



---

## Kernels

### Matrix Multiplication (`progress.cu`)

Progressively optimized kernels, benchmarked head-to-head against cuBLAS:

| Kernel |Description(Pseudo_name)| Strategy |
|---|---|---|
|mysgemm1| `naive` | Naive 2D thread per output element |
|mysgemm2| `coal` | Coalesced 1D thread indexing |
|mysgemm3| `tiled` | Shared memory tiling |
|mysgemm4| `tiled_1D` | 1D tiling + thread coarsening (TM output rows per thread) |
|mysgemm5| `tiled_2D_vec` | 2D tiling + register blocking (TM×TN) + `float4` vectorized loads, transposed A in SMEM |
|mysgemm6| `tiled_2D_vec_pad` | 2D tiling + register blocking (TM×TN) + `float4` vectorized loads, transposed A in SMEM  with padding (avoiding bankconflicts)|

### Matrix Multiplication (square, N×N)

> Tested on NVIDIA GeForce MX330 (Pascal Architecture) GPU. All times in ms, bandwidth in GB/s.
```
N= 256  naive:     0.5ms  coal:     0.5ms     tiled:     0.2ms   tiled_1d:     0.4ms  tiled_2d_vec:     0.1ms  tiled_2d_vec_pad:     0.1ms      cuBLAS:     0.1ms
N= 512  naive:     4.0ms  coal:     4.0ms     tiled:     1.7ms   tiled_1d:     2.1ms  tiled_2d_vec:     0.5ms  tiled_2d_vec_pad:     0.6ms      cuBLAS:     0.3ms
N=1024  naive:    29.9ms  coal:    31.5ms     tiled:    12.7ms   tiled_1d:    15.5ms  tiled_2d_vec:     3.4ms  tiled_2d_vec_pad:     3.7ms      cuBLAS:     2.1ms
N=2048  naive:   248.7ms  coal:   262.1ms     tiled:   100.3ms   tiled_1d:   122.1ms  tiled_2d_vec:    26.2ms  tiled_2d_vec_pad:    29.4ms      cuBLAS:    18.1ms
N=4096  naive:  2186.2ms  coal:  2144.5ms     tiled:   888.7ms   tiled_1d:  1075.5ms  tiled_2d_vec:   214.5ms  tiled_2d_vec_pad:   223.1ms      cuBLAS:   145.8ms
```

### Softmax (`softmax_bench.cu`, `softmax_playground.cuh`)

| Kernel | Strategy |
|---|---|
| `softmax_naive` | One thread per row, single-pass with global memory |
| `softmax_shared` | Online (numerically stable) softmax with shared memory tiling — single-pass max+denominator update |
| `softmax_warp_shfll` | Online softmax with register file usage|

### Softmax Benchmarks
```
shape (RxC)         naive (ms)    shared (ms)   warp shfl (ms)
---------------------------------------------------------------
4096 x 4096          73.405        21.570        5.504       
2048 x 8192          28.411        22.285        5.525       
 512 x 1024          0.926         0.777         0.180       
4096 x 16192         485.382       87.084        24.063 
```


### MNIST Neural Network (`nn.cu`)

A complete 3-layer MLP trained on MNIST written entirely in CUDA, this was to practice kernel fusion

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




## Auto-Tuner (`tune.sh`)

Sweeps the tiling parameter space for `mysgemm6` and finds the fastest valid configuration.

**Parameters swept:**

| Parameter | Values | Role |
|---|---|---|
| `BM`, `BN` | 32, 64, 128 | Output tile per CTA |
| `BK` | 8, 16, 32 | Phase depth along K |
| `TM`, `TN` |  8 | Per-thread register tile |
| `BLOCK_SIZE` | 32, 64, 128| Loading granularity (controls stride_A/stride_B) |
| `EXTRA_COLS` | 4, 8| For padding in SMEM|


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
