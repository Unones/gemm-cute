import torch
import cutlass
import cutlass.cute as cute
from math import prod

from cutlass.cute.runtime import from_dlpack


@cute.kernel
def _kernel_gemm_v0(
    mA : cute.Tensor,
    mB : cute.Tensor,
    mC : cute.Tensor,
    mD : cute.Tensor,
    tiled_mma : cute.TiledMma,
    atom_copy_ab : cute.CopyAtom,
    atom_copy_cd : cute.CopyAtom,
    nb_tiles_k : cutlass.Constexpr,
):
    """
    Perform a GEMM operation : D = A*B + C.
    
    Parameters
    ----------
    mA
        The first tensor of shape (M, K).
    mB
        The second tensor of shape (N, K).
    mC
        The third tensor of shape (M, N).
    mD
        The result tensor of shape (M, N).
    tiled_mma
        The TiledMma operation to execute the operation
        on the tensor cores.
    
    """
    
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    
    tile_cd = ((None, None), (bidx, bidy))
    
    thr_mma = tiled_mma.get_slice(tidx)
    
    gC = mC[tile_cd]    # (BS_M, BS_N)
    gD = mD[tile_cd]    # (BS_M, BS_N)
    
    tCgC = thr_mma.partition_C(gC)  # shape ((V_M, V_N), MMA_M, MMA_N)
    tCgD = thr_mma.partition_C(gD)  # shape ((V_M, V_N), MMA_M, MMA_N)
    
    tCrC = cute.make_rmem_tensor_like(tCgC, mC.dtype)
    cute.copy(atom_copy_cd, tCgC, tCrC,)
    
    rAcc = cute.make_rmem_tensor_like(tCgC, cutlass.Float32)
    rAcc.fill(0.0)
    
    rAcc.store(tCrC.load().to(cutlass.Float32))
    
    ## Creating once the registers for A and B
    tile_a = ((None, None), (bidx, 0))
    tile_b = ((None, None), (bidy, 0))
    
    gA = mA[tile_a]
    gB = mB[tile_b]
    
    tCgA = thr_mma.partition_A(gA)
    tCgB = thr_mma.partition_B(gB)
    
    tCrA = cute.make_rmem_tensor_like(tCgA, mA.dtype)
    tCrB = cute.make_rmem_tensor_like(tCgB, mB.dtype)
    ##
    
    for k in cutlass.range(nb_tiles_k):
        tile_a = ((None, None), (bidx, k))
        tile_b = ((None, None), (bidy, k))
        
        gA = mA[tile_a]
        gB = mB[tile_b]
        
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)

        cute.copy(atom_copy_ab, tCgA, tCrA,)
        cute.copy(atom_copy_ab, tCgB, tCrB,)
        
        cute.gemm(tiled_mma, rAcc, tCrA, tCrB, rAcc)
    
    tCrC.store(rAcc.load().to(mC.dtype))
    
    cute.copy(atom_copy_cd, tCrC, tCgD)



@cute.jit
def _kernel_host_gemm_v0(
    mA : cute.Tensor,
    mB : cute.Tensor,
    mC : cute.Tensor,
    mD : cute.Tensor,
):
    """
    Call the corresponding kernel for the gemm operation.
    
    Parameters
    ----------
    mA
        The first tensor of shape (M, K).
    mB
        The second tensor of shape (N, K).
    mC
        The third tensor of shape (M, N).
    mD
        The result tensor of shape (M, N).
    
    """
    _kernel_gemm_v0.set_name_prefix(
        "kernel_gemm_v0b",
        remove_cutlass_symbol=True,
        keep_mangled_name=False,
    )
    
    ## Set up the tiled MMA (for tensors mA and mB)
    shape_mnk = (16, 8, 16)
    op_mma = cute.nvgpu.warp.MmaF16BF16Op(
        mA.dtype,
        cutlass.Float32,
        shape_mnk,
    )
    
    atom_layout_mnk = (2, 2, 1)
    permutation_mnk = (1, 1, 1)
    
    tiled_mma = cute.make_tiled_mma(
        op_mma,
        atom_layout_mnk,
        permutation_mnk,
    )
    ##


    ## Set up atom copy for all tensors (no vectorization)
    op_copy = cute.nvgpu.CopyUniversalOp()
    atom_copy_ab = cute.make_copy_atom(
        op_copy,
        mA.dtype,
    )
    
    atom_copy_cd = cute.make_copy_atom(
        op_copy,
        mC.dtype,
    )
    ##
    
    M = cute.size(mA, mode=[0])
    K = cute.size(mA, mode=[1])
    N = cute.size(mC, mode=[1])
    
    ## Initializing coordinate tensors and using use of zipped divide
    size_tile_m = shape_mnk[0] * atom_layout_mnk[0] # BS_M
    size_tile_n = shape_mnk[1] * atom_layout_mnk[1] # BS_N
    size_tile_k = shape_mnk[2] * atom_layout_mnk[2] # BS_K
    
    tiler_mk = (size_tile_m, size_tile_k)   # (BS_M, BS_K)
    tiler_nk = (size_tile_n, size_tile_k)   # (BS_N, BS_K)
    tiler_mn = (size_tile_m, size_tile_n)   # (BS_M, BS_N)
    
    mA = cute.zipped_divide(mA, tiler_mk)   # ((BS_M, BS_K), (M / BS_M, K / BS_K))
    mB = cute.zipped_divide(mB, tiler_nk)   # ((BS_N, BS_K), (N / BS_N, K / BS_K))
    mC = cute.zipped_divide(mC, tiler_mn)   # ((BS_M, BS_N), (M / BS_M, N / BS_N))
    mD = cute.zipped_divide(mD, tiler_mn)   # ((BS_M, BS_N), (M / BS_M, N / BS_N))
    ##
    

    ## Initializing grid and launching kernel
    nb_warps_per_tile = prod(atom_layout_mnk)
    nb_threads_per_tile = 32 * nb_warps_per_tile
    
    grid_m = cute.ceil_div(M, size_tile_m)
    grid_n = cute.ceil_div(N, size_tile_n)
    nb_tiles_k = cute.ceil_div(K, size_tile_k)
    
    args = (mA, mB, mC, mD,
            tiled_mma,
            atom_copy_ab,
            atom_copy_cd,
            nb_tiles_k,
        )
    
    _kernel_gemm_v0(*args).launch(
        block=[nb_threads_per_tile, 1, 1],
        grid=[grid_m, grid_n, 1],
    )
    
    



def _check_device(device):
    """
    Check if the device given in input respects all conditions
    for CuTe DSL.
    
    Parameters
    ----------
    device
        The input device.
    
    Raises
    ------
    ValueError
        If the device given in input is not a CUDA device
        of Compute Capability above 8.0
    """
    
    if not torch.cuda.is_available():
        raise ValueError(f"CUDA is not available.")
    
    if device.type != "cuda":
        raise ValueError(f"The device given as input is not a CUDA device.")
    
    major, minor = torch.cuda.get_device_capability(device)
    
    if major < 8:
        raise ValueError(f"The Compute Capability of the CUDA device must be above 8. Got ({major},{minor}).")



def _check_tensor_dims(
    a : torch.Tensor,
    b : torch.Tensor,
    c : torch.Tensor,
):
    """
    Check if the input tensors have the correct dimensions.
    
    """
    
    if (a.ndim != 2) or (b.ndim != 2) or (c.ndim != 2):
        raise ValueError(f"Input tensors must be 2-dimensional. Got {a.shape}, {b.shape} and {c.shape}")
    
    M_A, N_A = a.shape
    M_B, N_B = b.shape
    M_C, N_C = c.shape
    
    if M_A != M_C:
        raise ValueError(f"The row dimension of a and c are not the same. Got {M_A} and {M_C}.")
    
    if N_A != M_B:
        raise ValueError(f"The contracting dimension of a and b are not the same. Got {M_B} and {N_A}.")
    
    if N_B != N_C:
        raise ValueError(f"The column dimension of b and c are not the same. Got {N_B} and {N_C}.")


    
    
def gemm_v0(
    a : torch.Tensor,
    b : torch.Tensor,
    c : torch.Tensor,
) -> torch.Tensor:
    """
    Call the kernel _kernel_gemm_v0 to perform the simplest GEMM possible.
    
    The executed operation is : D = A*B + C.
    
    Parameters
    ----------
    a
        The first tensor, shape (M, K).
    b
        The second tensor, shape (K, N).
    c
        The third tensor, shape (M, N).
    
    Returns
    -------
    d
        The result tensor, shape (M, N).
    
    Raises
    ------
    ValueError
        t
    """
    
    ## Sanity checks
    _check_tensor_dims(a, b, c)
    
    if not (a.device == b.device and b.device == c.device):
        raise ValueError(f"All tensors must be on the same device.")
    
    if (a.dtype != b.dtype):
        raise ValueError(f"A and B must have the same dtype. Got {a.dtype} and {b.dtype}.")
    if a.dtype != torch.bfloat16:
        raise ValueError(f"A and B must have the dtype torch.bfloat16.")
    
    device = a.device
    
    _check_device(device)
    ##
    
    b = b.permute(1, 0)     # Get B into a K-major tensor for the MMA
    d = torch.empty(c.shape, dtype=c.dtype, device=device)
    
    a_ = from_dlpack(a)
    b_ = from_dlpack(b)
    c_ = from_dlpack(c)
    d_ = from_dlpack(d)
    
    _kernel_host_gemm_v0(a_, b_, c_, d_)
    
    return d
    
    
    
    
if __name__ == "__main__":
    M = 2048
    N = 2048
    K = 2048
    
    torch.manual_seed(43)
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    a = torch.randn((M, K), dtype=dtype, device=device)
    b = torch.randn((K, N), dtype=dtype, device=device)
    c = torch.randn((M, N), dtype=dtype, device=device)
    
    d = gemm_v0(a, b, c)
    
    d_test = (a.float()@b.float() + c.float()).to(dtype=dtype)
    
    # print(f"The output calculated by the kernel is equal to : \n{d}")
    # print(f"The output calculated by PyTorch is equal to : \n{d_test}")
    
    torch.testing.assert_close(d, d_test, atol=1e-2, rtol=1e-2)