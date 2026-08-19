# I/ Results


# II/ The most basic GEMM in CuTe DSL

## A) Choices made

For the first GEMM kernel, I decided that I would use as less APIs as possible to
keep the kernel and host function as basic as possible.
I did not use any `tiled_copy` nor any `predicate tensors`. I did not use many 
optimisations available in CuTe so as to understand the basics of a GEMM.

My first kernel (v0) does the following :
- load elements from the global memory to the registers (A, B and C)
- compute the GEMM using `cute.gemm` with the accumulator living in the registers
- store elements from the registers to the global memory

There are no `swizzling`, use of `shared memory` and whatever else.

From this basic kernel, I will implement in future version of this GEMM kernel 
predicates, use of shared memory, swizzling, ...

## B) Memory or Compute Bound ?

In every kernel, the ridge point of the `Arithmetic Intensity` will be kept the same.
It is certain that every kernel will have a unique ridge point depending on the 
transfers from global memory to another part of the memory in the GPU (L2, L1, registers).

However, this base ridge point will serve as a minimum to know approximately where the kernels
will transition from being memory-bound to being compute-bound.

The definition of the `Arithmetic Intensity` is the following : 
` AI = FLOPs / Bytes transferred from memory`

The total number of FLOPs from the matrix multiplication is equal to : `2 * M * N * K`.
The total number of bytes transferred from memory (with the dtype `bf16`) is equal to : 
`2*M*N + 2*M*K + 2*N*K`.

All tests and profilings will be done on square matrices for simplicity. Therefore : 
`M = N = K`.
Consequently, `AI = (2 * M*M*M) / (6 * M*M)`.
Therefore, `AI = M / 3`.

All compute is done on a RTX 5070 Ti locked at 2.30 GHz. Therefore, the number of TFLOPs
using tensor cores on BF16-input/FP32-accumulation is equal to `82.5 TFLOP/s`. 
This GPU has a bandwidth of `896 GB/s`.
Therefore, the ridge point in this configuration is equal to : `92 FLOPs / byte`.

Using the calculation leading to `AI`, a GEMM kernel is memory-bound below `M = 276`
and compute-bound above `M = 276`.

All sizes in benchmarks wil be powers of 2. Therefore, it is expected to have a kernel
fully compute-bound at `M = 512` and above.
To conclude this section, any kernel is supposed to be :
- memory-bound at `M = 256` and below
- compute-bound at `M = 512` and above

## C) Decicions made in the kernel

The biggest decision that had to be taken is regarding the data types.

All tensors are in `bf16`. According to the `CuTe DSL` documentation, with such input,
the accumulator must be in `fp32`.
I had the possibility to deal with it in two ways.

Either I initialize a tensor in registers with the `fp32` dtype then make every element 
equals to 0 then use it in an accumulator of the cute.gemm. At the time of adding the 
elements from `tCrC`, a certain amount of registers would be alive.
This would work but is not optimized in the way registers are used.

``` python
rAcc = cute.make_rmem_tensor_like(tCgC, cutlass.Float32)
rAcc.fill(0.0)
...
cute.gemm(tiled_mma, rAcc, tCrA, tCrB, rAcc)
```

The other way to do it is to promote the rmem tensor after the copy. Indeed, the atom copy
instanciated in the host code forces the same dtype on both ends.
The code is the following :

``` python
tCrC = cute.make_rmem_tensor_like(tCgC, mC.dtype)
cute.copy(atom_copy_cd, tCgC, tCrC,)

tCrC_f32 = cute.make_rmem_tensor_like(tCgC, cutlass.Float32)
tCrC_f32.store(tCrC.load().to(cutlass.Float32))
...
cute.gemm(tiled_mma, tCrC_f32, tCrA, tCrB, tCrC_f32)
...
tCrC.store(tCrC_f32.load().to(mC.dtype))
```

This allows for maximum precision with the FP32-accumulator and the least amount of registers
used. Despite being a Blackwell architecture, the `RTX 5070 Ti` does not have a tensor memory 
to store the accumulator.

## D) Benchmark

<img src="benchmarks/figures/benchmark_gemm_v0.png" alt="Comparison kernel_gemm_v0 vs Pytorch on RTX 5070 Ti" width="700">

The benchmark produces the following results with square matrices (clock at `2.30 GHz`):

| Metric | Value Custom kernel | Value PyTorch |
|---|---|---|
| Throughput (M = 64) | ~0.07 TFLOP/s | ~0.03 TFLOP/s |
| Throughput (M = 128) | ~0.6 TFLOP/s | ~0.26 TFLOP/s |
| Throughput (M = 256) | ~4.8 TFLOP/s | ~2.1 TFLOP/s |
| Throughput (M = 512) | ~6.0 TFLOP/s | ~17 TFLOP/s |
| Throughput (M = 1024) | ~6.5 TFLOP/s | ~58 TFLOP/s |
| Throughput (M = 2048) | ~6.7 TFLOP/s | ~76 TFLOP/s |
| Throughput (M = 4096) | ~7.0 TFLOP/s | ~79 TFLOP/s |

The maximum compute on this GPU at this locked clock is equal to : `82.5 TFLOP/s`.
My first GEMM kernel is much lower than the kernel from PyTorch.
It was to be expected that my kernel would have such low throughput. Let's launch a profiling
to understand what went wrong exactly.


## E) Nsight Compute Report

The profiled shape is `M = N = K = 2048`. The GPU clock is fixed at `2.30 GHz`.

The Speed of Light report gives us this information : 

| Metric | Value |
|---|---|
| Compute (SM) throughput | 25.92% |
| Memory throughput | 93.79% |
| L1 Cache Throughput | 87.67% |
| L2 Cache Throughput | 93.79% |
| DRAM throughput | 1.18% |

These metrics already give us plenty of information to understand why my kernel is so slow.

As proven above, with the `Arithmetic Intensity`, this kernel should be compute-bound, meaning
that the Compute Throughput should be at least above 70% or 80%. 
However, in this situation, we can see that it is under 26%. Furthermore, the DRAM throughput is
at less than 2%. Therefore, the kernel is not memory-bound.

If a kernel is neither memory-bound nor compute-bound, there are high chances that it is latency-
bound. This means that the threads (and warps as the MMA insruction is issued with `wmma`) wait 
a huge part of their time waiting for the data to be in the registers.

This hypothesis can be answered in the NCU report in the `Scheduler Statistics` section. The metric
to look out for is the `No Elligible` to know the percentage of warps that are waiting idle.
This metric is at `94.08 %`, which is a considerable amount of time where the warps do absolutely 
nothing.

Therefore, our kernel is latency-bound because not enough data to feed the tensor cores.
But one question remains : why is there not enough data while the DRAM transfers little data over 
time ?
The answer may be in the high throughputs of the L1 and L2 caches. 

But construction of the kernel, there is no coalescing to get the data from the global memory.
All copies follow the same principle in the kernel : 

```python
tile_cd = ((None, None), (bidx, bidy))
gC = mC[tile_cd]
tCgC = thr_mma.partition_C(gC)
tCrC = cute.make_rmem_tensor_like(tCgC, mC.dtype)
cute.copy(atom_copy_cd, tCgC, tCrC,)
```

This means that I use the `ThrMma` class instead of using the `ThrCopy` class. The former has a
specific thread-value layout adapted for the Tensor Cores. The latter can be initialized to have 
the best thread-value layout to maximize coalescing.

Therefore, in each loop : 
``` python
for k in cutlass.range(nb_tiles_k):
    ...
    tCgA = thr_mma.partition_A(gA)
    tCgB = thr_mma.partition_B(gB)

    cute.copy(atom_copy_ab, tCgA, tCrA,)
    cute.copy(atom_copy_ab, tCgB, tCrB,)
```

We can see here that the uncoalesced TV layout from `thr_mma` is used to copy the elements from A and B
(the same problem stands for the tensor C).
Therefore, for each and every iteration of the loop, new elements are copied from the global memory to the
registers. I did not put into place any strategies to store in the `shared memory` elements that are used
in the tile. 
However, the compiler still tries to do its best. Indeed, L1 and L2 caches have an extremely high throughput to
try and minimize the number of transfers from the global memory to the registers.

Furthermore, no vectorization has been put in place. Consequently, half of the lines in the SASS code are 
load instructions from the global memory. Here is one example : 
```python
LDG.E.U16 R14, desc[UR6][R22.64]
```

There is also another pice of advice from the SASS code. Next to each `LDG` instruction, there is marked :
`75% of this line's global accesses are excessive`. This comes from the fact that only `16 bits` are used while
the `LDG` instruction fetches `64 bits`. This is another problem to tackle.


## F) Planned Improvements

From what was found in the NCU Report, my first GEMM kernel can benefit from 2 major improvements :
- using the shared memory to load once all the elements from the tile and reduce consequently all transfer costs
- using the vectorization up to 128 bits

