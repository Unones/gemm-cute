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