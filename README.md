# CUDA Matrix Multiplication Optimization

A CUDA C/C++ project that implements and compares naive matrix multiplication with a shared-memory tiled implementation. The project explores GPU parallelism, memory reuse, kernel timing, correctness verification, and per-thread instrumentation.

## Overview

Given two square matrices \(A\) and \(B\), each of dimension \(N X N\), both kernels compute the matrix product \(C = AB\):

$$C_{ij} = \sum_{k=0}^{N-1} A_{ik} B_{kj}
$$
The project includes two implementations:

- **Naive kernel (`matmul`):** Each CUDA thread computes one output element by reading the corresponding row of \(A\) and column of \(B\) directly from global memory.
- **Tiled kernel (`matmulTiled`):** Each thread block cooperatively loads tiles of \(A\) and \(B\) into shared memory, reusing loaded values across multiple output calculations.

The tiled version is designed to reduce redundant global-memory requests and improve data reuse.

## Implementations

### 1. Naive matrix multiplication

Each thread computes one element of the output matrix. For every value of \(k\), it loads `A[row * N + k]` and `B[k * N + col]`, then accumulates their product.

**Characteristics:**

- Straightforward baseline implementation.
- Uses a \(16 \times 16\) thread block.
- Performs \(N\) multiply-accumulate iterations per output element.
- Records per-thread logical operation counts and elapsed clock cycles.
- Writes per-thread statistics to `thread_stats.csv` and a run summary to `summary.txt`.

> The source-level logical memory counts are not measurements of actual DRAM traffic. Caching, compiler optimization, and memory transactions affect physical traffic.

### 2. Shared-memory tiled matrix multiplication

The tiled kernel divides the matrices into `TILE × TILE` tiles. Threads cooperatively load tiles into shared memory, synchronize, and reuse the loaded values to compute partial sums.

**Characteristics:**

- Uses shared memory for tile reuse.
- Uses `__syncthreads()` to coordinate tile loading and computation.
- Handles dimensions that are not multiples of `TILE` by zero-padding out-of-bounds loads.
- Allows `TILE` to be configured at compile time; the default is `16`.
- Runs warm-up iterations before timed repetitions.
- Verifies output values against the expected result.

## Repository layout

Adapt the filenames below to match the names used in your repository.

```text
.
├── matmul.cu          # Naive kernel and per-thread instrumentation
├── matmul_tiled.cu    # Shared-memory tiled kernel
└── README.md
```

The naive executable generates these output files in its working directory:

- `thread_stats.csv`
- `summary.txt`

## Requirements

- NVIDIA GPU with CUDA support
- NVIDIA CUDA Toolkit (`nvcc`)
- C++ compiler supported by your CUDA Toolkit

Check that the compiler and GPU are available:

```bash
nvcc --version
nvidia-smi
```

## Build

Compile the naive implementation:

```bash
nvcc -O2 matmul.cu -o matmul
```

Compile the tiled implementation:

```bash
nvcc -O2 matmul_tiled.cu -o matmul_tiled
```

To build the tiled kernel with a different tile size, for example `32`:

```bash
nvcc -O2 -DTILE=32 matmul_tiled.cu -o matmul_tiled_32
```

Choose a tile size supported by the target GPU and ensure the resulting block size does not exceed the device's threads-per-block limit. Tile size can affect register usage, shared-memory usage, occupancy, and performance.

## Run

### Naive implementation

```bash
./matmul 256
```

The argument is the matrix dimension `N`; the default is `256`. The program currently accepts values from `1` through `2048`.

The program writes the following output files:

- **`thread_stats.csv`:** One row per output element/thread, including grid and block coordinates, logical operation counts, logical memory counts, and the measured clock-cycle interval.
- **`summary.txt`:** Matrix size, launch dimensions, CUDA-event kernel time, minimum/maximum/mean per-thread cycle counts, total algorithmic floating-point operation count, and estimated throughput.

### Tiled implementation

```bash
./matmul_tiled 512 20
```

Arguments:

- `N`: Matrix dimension (default: `512`).
- `repeats`: Number of timed kernel executions (default: `20`).

The program performs five warm-up launches, times the requested number of launches, reports the average kernel time and estimated GFLOP/s, and checks the result.

## Correctness verification

The supplied programs initialize their matrices to constant values, making the expected results easy to verify:

- **Naive implementation:** \(A[i] = 1.0\), \(B[i] = 2.0\), so each output element should equal \(2N\).
- **Tiled implementation:** \(A[i] = 1.0\), \(B[i] = 1.0\), so each output element should equal \(N\).

The tiled implementation checks every output element using a floating-point tolerance. The naive implementation checks its outputs against its expected constant result.

These checks are useful sanity tests, but constant-valued inputs do not cover every indexing or numerical edge case. More comprehensive testing should also use nonuniform input matrices and compare the results against a CPU reference implementation.

## Benchmarking methodology

Both programs use CUDA events to measure kernel execution time. The tiled version also runs warm-up iterations and averages timing over multiple launches.

The programs calculate approximate algorithmic throughput as:

\[
\text{GFLOP/s} = \frac{2N^3}{t_{\text{seconds}} \times 10^9}
\]

Here, \(2N^3\) is the conventional operation count for dense matrix multiplication, counting one multiplication and one addition per inner-loop iteration. Actual instruction counts may differ because compilers can fuse multiply-add operations.

For fair comparisons:

1. Use the same GPU, matrix dimensions, compiler settings, and comparable measurement procedures.
2. Warm up both implementations before timing.
3. Run multiple repetitions and compare average times.
4. Verify correctness for every tested configuration.
5. Record the GPU model, CUDA version, compiler flags, tile size, and matrix dimensions.
6. Use profiling tools such as NVIDIA Nsight Compute to investigate memory throughput, cache behavior, occupancy, and warp-stall reasons.

**Important:** The current implementations use different initialization values and timing procedures. For a rigorous direct comparison, align the input data, warm-up procedure, repetitions, and validation logic.

## Understanding the per-thread statistics

The naive kernel records the following fields in `thread_stats.csv`:

| Field | Meaning |
|---|---|
| `row`, `col` | Output matrix element computed by the thread |
| `block_x`, `block_y` | CUDA block coordinates |
| `thread_x`, `thread_y` | Thread coordinates within the block |
| `multiplications` | Source-level multiplication count |
| `additions` | Source-level accumulation count |
| `logical_loads` | Source-level input load count |
| `logical_stores` | Source-level output store count |
| `logical_bytes` | Estimated bytes from logical load/store counts, assuming four bytes per `float` |
| `elapsed_clock_cycles` | Difference between `clock64()` readings around the thread's computation and store |

Per-thread clock-cycle measurements include effects such as stalls and scheduling, and can be affected by instrumentation overhead. They should not be summed to estimate total kernel duration or treated as isolated instruction latency. The logical memory counts do not measure actual DRAM bytes transferred.

## Limitations and possible extensions

- Use nonuniform matrices and a CPU reference for stronger correctness tests.
- Compare multiple tile sizes, such as `8`, `16`, and `32`, where supported.
- Benchmark across several matrix dimensions.
- Use Nsight Compute to measure hardware performance counters instead of inferring physical memory behavior from source-level counts.
- Consider a higher-performance implementation using register tiling, vectorized memory accesses, or CUDA libraries such as cuBLAS.
- Report repeatable measurements with GPU and software environment details.

## Concepts demonstrated

- CUDA thread and block indexing
- GPU kernel launches
- Global memory and shared memory
- Shared-memory tiling and data reuse
- Thread synchronization
- CUDA event-based timing
- Correctness verification
- Per-thread instrumentation
- Performance benchmarking and GFLOP/s estimation
