import torch
import cutlass
import cutlass.cute as cute
from math import prod

from cutlass.cute.runtime import from_dlpack

@cute.kernel
def _kernel_gemm_v1(
    mA : cute.Tensor,
    mB : cute.Tensor,
    mC : cute.Tensor,
    mD : cute.Tensor,
    tiled_mma : cute.TiledMma,
    tiled_copy_a : cute.TiledCopy,
    tiled_copy_b : cute.TiledCopy,
    tiled_copy_cd : cute.TiledCopy,
    copy_atom_mk : cute.CopyAtom,
    copy_atom_nk : cute.CopyAtom,
    copy_atom_mn : cute.CopyAtom,
    nb_tiles_k : cutlass.Constexpr,
    bs_m : cutlass.Constexpr,
    bs_n : cutlass.Constexpr,
    bs_k : cutlass.Constexpr,
):
    """
    
    """
    
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    
    thr_mma = tiled_mma.get_slice(tidx)
    thr_copy_a = tiled_copy_a.get_slice(tidx)
    thr_copy_b = tiled_copy_b.get_slice(tidx)
    thr_copy_cd = tiled_copy_cd.get_slice(tidx)
    
    ## Set up shared memory
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(
        mA.dtype,
        cute.make_layout((bs_m, bs_k), stride=(bs_k, 1)),
        byte_alignment=16,
    )
    sB = smem.allocate_tensor(
        mB.dtype,
        cute.make_layout((bs_n, bs_k), stride=(bs_k, 1)),
        byte_alignment=16,
    )
    sC = smem.allocate_tensor(
        mC.dtype,
        cute.make_layout((bs_m, bs_n), stride=(bs_n, 1)),
        byte_alignment=16,
    )
    ##
    
    ## Load from global memory to shared memory
    tile_cd = ((None, None), (bidx, bidy))
    gC = mC[tile_cd]
    tAgC = thr_copy_cd.partition_S(gC)
    tAsC = thr_copy_cd.partition_S(sC)
    cute.copy(tiled_copy_cd, tAgC, tAsC)
    
    cute.arch.barrier()
    
    tCsC = thr_mma.partition_C(sC)
    tCrC = cute.make_rmem_tensor_like(tCsC, mC.dtype)
    cute.copy(copy_atom_mn, tCsC, tCrC)
    
    rAcc = cute.make_rmem_tensor_like(tCsC, cutlass.Float32)
    rAcc.fill(0.0)
    
    rAcc.store(tCrC.load().to(cutlass.Float32))
    
    for k in cutlass.range(nb_tiles_k):
        tile_a = ((None, None), (bidx, k))
        gA = mA[tile_a]
        tAgA = thr_copy_a.partition_S(gA)
        tAsA = thr_copy_a.partition_D(sA)
        cute.copy(tiled_copy_a, tAgA, tAsA)
        
        tile_b = ((None, None), (bidy, k))
        gB = mB[tile_b]
        tBgB = thr_copy_b.partition_S(gB)
        tBsB = thr_copy_b.partition_D(sB)
        cute.copy(tiled_copy_b, tBgB, tBsB)
        
        cute.arch.barrier()
    
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        
        tCrA = cute.make_rmem_tensor_like(tCsA, mA.dtype)
        tCrB = cute.make_rmem_tensor_like(tCsB, mB.dtype)
        
        cute.copy(copy_atom_mk, tCsA, tCrA)
        cute.copy(copy_atom_nk, tCsB, tCrB)
        
        cute.gemm(tiled_mma, rAcc, tCrA, tCrB, rAcc)
        
        cute.arch.barrier()

    tCrC.store(rAcc.load().to(mD.dtype))
    ##
    
    ## Store result back into global memory
    cute.copy(copy_atom_mn, tCrC, tCsC)
    
    cute.arch.barrier()
    
    gD = mD[tile_cd]
    
    tAsD = thr_copy_cd.partition_S(sC)
    tAgD = thr_copy_cd.partition_D(gD)
    
    cute.copy(tiled_copy_cd, tAsD, tAgD)
    
    

@cute.jit
def _kernel_host_gemm_v1(
    mA : cute.Tensor,
    mB : cute.Tensor,
    mC : cute.Tensor,
    mD : cute.Tensor,
):
    """
    
    
    """
    
    _kernel_gemm_v1.set_name_prefix(
        "kernel_gemm_v1",
        remove_cutlass_symbol=True,
        keep_mangled_name=False,
    )
    
    ## Setup up the tiled MMA
    shape_mnk = (16, 8, 16)
    
    op_mma = cute.nvgpu.warp.MmaF16BF16Op(
        mA.dtype,
        cutlass.Float32,
        shape_mnk=shape_mnk,
    )
    
    atom_layout_mnk = (2, 2, 1)
    permutation_mnk = (1, 1, 1)
    
    tiled_mma = cute.make_tiled_mma(
        op_mma,
        atom_layout_mnk,
        permutation_mnk,
    )
    ##
    
    ## Set up the tiled copy
    bs_m = shape_mnk[0] * atom_layout_mnk[0]
    bs_n = shape_mnk[1] * atom_layout_mnk[1]
    bs_k = shape_mnk[2] * atom_layout_mnk[2]
    
    max_nb_elems_per_thr = 8    # vectorization bf16
    
    nb_elems_mk = bs_m * bs_k
    nb_elems_nk = bs_n * bs_k
    nb_elems_mn = bs_m * bs_n
    
    nb_warps_per_block = prod(atom_layout_mnk)
    nb_theads_per_block = 32 * nb_warps_per_block   # 32 threads per warp
    
    nb_elems_per_thr_mk = min(nb_elems_mk // nb_theads_per_block, max_nb_elems_per_thr)
    nb_elems_per_thr_nk = min(nb_elems_nk // nb_theads_per_block, max_nb_elems_per_thr)
    nb_elems_per_thr_mn = min(nb_elems_mn // nb_theads_per_block, max_nb_elems_per_thr)
    
    
    val_layout_mk = cute.make_layout((1, nb_elems_per_thr_mk), stride=(0, 1))
    val_layout_nk = cute.make_layout((1, nb_elems_per_thr_nk), stride=(0, 1))
    val_layout_mn = cute.make_layout((1, nb_elems_per_thr_mn), stride=(0, 1))
    thr_layout_mk = cute.make_layout((bs_m, bs_k//nb_elems_per_thr_mk), stride=(bs_k//nb_elems_per_thr_mk, 1))
    thr_layout_nk = cute.make_layout((bs_n, bs_k // nb_elems_per_thr_nk), stride=(bs_k // nb_elems_per_thr_nk, 1))
    thr_layout_mn = cute.make_layout((bs_m, bs_n // nb_elems_per_thr_mn), stride=(bs_n // nb_elems_per_thr_mn, 1))
    
    op_copy = cute.nvgpu.CopyUniversalOp()
    copy_atom_mk = cute.make_copy_atom(
        op_copy,
        mA.dtype,
        # num_bits_per_copy=32,
    )
    copy_atom_nk = cute.make_copy_atom(
        op_copy,
        mA.dtype,
        # num_bits_per_copy=32,
    )
    copy_atom_mn = cute.make_copy_atom(
        op_copy,
        mA.dtype,
        # num_bits_per_copy=32,
    )
    
    tiler_mk, layout_tv_mk = cute.make_layout_tv(
        thr_layout_mk,
        val_layout_mk,
    )
    tiler_nk, layout_tv_nk = cute.make_layout_tv(
        thr_layout_nk,
        val_layout_nk,
    )
    tiler_mn, layout_tv_mn = cute.make_layout_tv(
        thr_layout_mn,
        val_layout_mn,
    )
    
    tiled_copy_mk = cute.make_tiled_copy(
        copy_atom_mk,
        layout_tv_mk,
        tiler_mk,
    )
    tiled_copy_nk = cute.make_tiled_copy(
        copy_atom_nk,
        layout_tv_nk,
        tiler_nk,
    )
    tiled_copy_mn = cute.make_tiled_copy(
        copy_atom_mn,
        layout_tv_mn,
        tiler_mn,
    )
    ##
    
    M = cute.size(mD, mode=[0])
    N = cute.size(mD, mode=[1])
    K = cute.size(mA, mode=[1])
    
    ## Set up the tensors
    mA = cute.zipped_divide(mA, tiler_mk)
    mB = cute.zipped_divide(mB, tiler_nk)
    mC = cute.zipped_divide(mC, tiler_mn)
    mD = cute.zipped_divide(mD, tiler_mn)
    ##
    
    ## Initializing the grid and launching the grid
    grid_m = cute.ceil_div(M, bs_m)
    grid_n = cute.ceil_div(N, bs_n)
    nb_tiles_k = cute.ceil_div(K, bs_k)
    
    args = (
        mA, mB, mC, mD,
        tiled_mma,
        tiled_copy_mk,
        tiled_copy_nk,
        tiled_copy_mn,
        copy_atom_mk,
        copy_atom_nk,
        copy_atom_mn,
        nb_tiles_k,
        bs_m, bs_n, bs_k,
    )
    
    _kernel_gemm_v1(*args).launch(
        block=[nb_theads_per_block, 1, 1],
        grid=[grid_m, grid_n, 1],
    )
    
    
    
    
    

def gemm_v1(
    a : torch.Tensor,
    b : torch.Tensor,
    c : torch.Tensor,
) -> torch.Tensor:
    """
    
    """
    
    device = a.device
    
    b = b.permute(1, 0)     # Get B into a K-major tensor for the MMA
    d = torch.empty(c.shape, dtype=c.dtype, device=device)
    
    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)
    c_ = from_dlpack(c, assumed_align=16)
    d_ = from_dlpack(d, assumed_align=16)
    
    _kernel_host_gemm_v1(a_, b_, c_, d_)
    
    return d


if __name__ == "__main__":
    M = 256
    N = 256
    K = 256
    
    torch.manual_seed(43)
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    a = torch.randn((M, K), dtype=dtype, device=device)
    b = torch.randn((K, N), dtype=dtype, device=device)
    c = torch.randn((M, N), dtype=dtype, device=device)
    
    d = gemm_v1(a, b, c)
    
    d_test = (a.float()@b.float() + c.float()).to(dtype=dtype)
    
    # print(f"The output calculated by the kernel is equal to : \n{d}")
    # print(f"The output calculated by PyTorch is equal to : \n{d_test}")
    
    torch.testing.assert_close(d, d_test, atol=1e-2, rtol=1e-2)