# GPU-CUDA-Kernel

A progressive CUDA kernel repository covering CUDA fundamentals, optimization patterns, performance engineering, and AI kernel implementations.

The goal of this repository is to build an understanding of GPU programming, memory hierarchy, parallel execution, and deep learning operator implementation.

# Repository Structure

```text

CUDA-Kernels
│
├── 01_fundamentals
│   ├── vector_add.cu
│   ├── scalar_multiply.cu
│   ├── elementwise_multiply.cu
│   ├── relu.cu
│   ├── sigmoid.cu
│   └── grid_stride_loop.cu
│
├── 02_reduction_and_memory
│   ├── simple_reduction.cu
│   ├── atomic_reduction.cu
│   ├── shared_memory_reduction.cu
│   ├── matrix_transpose.cu
│   └── tiled_matmul.cu
│
├── 03_performance_engineering
│   ├── kernel_timer.cu
│   ├── memory_bandwidth.cu
│   ├── naive_vs_optimized_reduction.cu
│   └── occupancy_experiment.cu
│
└── 04_ai_kernels
    ├── gelu.cu
    ├── softmax.cu
    ├── fused_bias_relu.cu
    └── layernorm.cu

```

<!-- 1 -->

# 1 CUDA Fundamentals

This section introduces the basic CUDA programming model and elementwise GPU kernels.

## vector_add.cu

Vector addition is often the first CUDA kernel.

```cpp
C[i] = A[i] + B[i]
```
**CONCEPT**
- threadIdx
- blockIdx
- blockDim
- gridDim
- Global Memory Access

## scalar_multiply.cu

Multiply every element by a scalar value.

```cpp
C[i] = alpha * A[i]
```

**CONCEPT**
- Elementwise Parallelism
- Memory Throughput
- GPU Thread Mapping

## elementwise_multiply.cu

Multiply two vectors element by element.

```cpp
C[i] = A[i] * B[i]
```

**CONCEPT**
- Multiple Input Tensors
- Parallel Data Processing
- Memory Access Patterns


## relu.cu

Rectified Linear Unit activation.

```cpp
output[i] = max(0, input[i])
```

**CONCEPT**
- Activation Functions
- Branching in GPU Kernels
- AI Workloads

## sigmoid.cu

Sigmoid activation.

```cpp
output[i] = 1 / (1 + exp(-x))
```
**CONCEPT**
- Nonlinear Functions
- Mathematical Operators
- GPU Math Libraries

## grid_stride_loop.cu

Process datasets larger than the number of launched threads.

```cpp
for (int i = idx; i < N; i += stride)
```

**CONCEPT**
- Grid-Stride Loop
- Scalability
- Persistent Work Distribution


<!-- 2 -->

# 2 Optimization Patterns

This section introduces common GPU optimization patterns.

## simple_reduction.cu

Compute the sum of an array.

```cpp
sum += input[i]
```

**CONCEPT**
- Reduction Pattern
- Tree Reduction
- Parallel Aggregation

## atomic_reduction.cu

```cpp
atomicAdd()
```

**CONCEPT**
- Atomic Operations
- Synchronization
- Contention

## shared_memory_reduction.cu

Reduction using shared memory.

**CONCEPT**
- Shared Memory
- Synchronization
- Reduced Global Memory Traffic


## matrix_transpose.cu

Transpose a matrix efficiently on the GPU.

**CONCEPT**
- Memory Coalescing
- Shared Memory Tiles
- Bank Conflicts


## tiled_matmul.cu

Tiled matrix multiplication using shared memory.

```cpp
C = A × B
```

**CONCEPT**
- Matrix Multiplication
- Shared Memory Tiling
- Data Reuse
- Compute Intensity


<!-- 3 -->
# 3 Performance Engineering

This section focuses on performance measurement and optimization.

## kernel_timer.cu

Measure kernel execution time using CUDA events.

**CONCEPT**
- cudaEventRecord
- Kernel Latency
- Runtime Measurement

## memory_bandwidth.cu

Measure effective memory bandwidth.

**CONCEPT**
- Global Memory Throughput
- Effective Bandwidth
- Memory-Bound Kernels

## naive_vs_optimized_reduction.cu

Compare naive reduction with optimized reduction.

**CONCEPT**
- Optimization Analysis
- Shared Memory Benefits
- Performance Comparison


## occupancy_experiment.cu

Evaluate different thread-block configurations.

**CONCEPT**
- Occupancy
- Thread Scheduling
- Resource Utilization


<!-- 4 -->

# 4 AI Kernels

This section implements common deep learning operators.

## gelu.cu

Gaussian Error Linear Unit.

**CONCEPT**
- Nonlinear Activations
- Transformer Models

## softmax.cu

Softmax normalization.

**CONCEPT**
- Numerical Stability
- Probability Distribution

## fused_bias_relu.cu

Fuse bias addition and activation into one kernel.

**CONCEPT**
- Kernel Fusion
- Reduced Memory Traffic
- AI Inference Optimization

## layernorm.cu

Layer normalization.

**CONCEPT**
- Normalization
- Statistics on GPU
- Transformer Infrastructure



# Motivation

This repository documents my journey learning CUDA kernel development, GPU performance optimization, and AI infrastructure engineering.

The long-term goal is to develop expertise in GPU systems, high-performance computing, and AI infrastructure.









