import torch
import cutlass
import cutlass.cute as cute
from math import prod

from cutlass.cute.runtime import from_dlpack


@cute.kernel
def _kernel_gemm_v3(
    mA : cute.Tensor,
    mB : cute.Tensor,
    mC : cute.Tensor,
    mD : cute.Tensor,
    tiled_mma : cute.TiledMma,
    tiled_copy_async_mk : cute.TiledCopy,
    tiled_copy_async_nk : cute.TiledCopy,
    copy_atom_s2r : cute.CopyAtom,
    copy_atom_mn : cute.CopyAtom,
    nb_tiles_k : cutlass.Constexpr,
    bs_m : cutlass.Constexpr,
    bs_n : cutlass.Constexpr,
    bs_k : cutlass.Constexpr,
    num_stages : cutlass.Constexpr,
    
):
    """
    
    
    """
    
    ## Initiate needed variables
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    
    thr_copy_mk = tiled_copy_async_mk.get_slice(tidx)
    thr_copy_nk = tiled_copy_async_nk.get_slice(tidx)
    thr_mma = tiled_mma.get_slice(tidx)
    
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(
        mA.dtype,
        cute.make_ordered_layout((bs_m, num_stages*bs_k), order=(1, 0)),
        byte_alignment=16,
        # swizzle=cute.make_swizzle(1, 4, 3),
    )
    sB = smem.allocate_tensor(
        mB.dtype,
        cute.make_ordered_layout((bs_n, num_stages*bs_k), order=(1, 0)),
        byte_alignment=16,
        # swizzle=cute.make_swizzle(1, 4, 3),
    )
    
    sA = cute.logical_divide(sA, (None, bs_k))      # dividing sA in num_stages tensors (bs_m, bs_k)
    sB = cute.logical_divide(sB, (None, bs_k))      # same for sB
    ##
    
    ## Loading C elements to reegisters
    tile_cd = ((None, None), (bidx, bidy))
    
    gC = mC[tile_cd]
    tCgC = thr_mma.partition_C(gC)
    tCrC = cute.make_rmem_tensor_like(tCgC, mC.dtype)
    
    cute.copy(copy_atom_mn, tCgC, tCrC)
    
    rAcc = cute.make_rmem_tensor_like(tCrC, cutlass.Float32)
    rAcc.fill(0.0)
    rAcc.store(tCrC.load().to(cutlass.Float32))
    ##
    
    ## Prologue for async copy
    for i in cutlass.range(num_stages - 1):
        
        # if tidx==0 and bidx==0 and bidy==0:
        #     cute.printf("The buffer loaded is : %d", i)
        #     cute.printf("The buffer loaded in shared memory is at the location : %d", i%num_stages)
        
        tile_a = ((None, None), (bidx, i))
        tile_b = ((None, None), (bidy, i))
        
        gA = mA[tile_a]
        gB = mB[tile_b]
        
        tAgA = thr_copy_mk.partition_S(gA)
        tAgB = thr_copy_nk.partition_S(gB)
        
        tAsA = thr_copy_mk.partition_D(sA[(None, (None, i))])
        tAsB = thr_copy_nk.partition_D(sB[(None, (None, i))])
        
        # if tidx==0 and bidx==0 and bidy==0:
        #     cute.printf("The layout os tAgB is : {}", tAgB.layout)
        #     cute.printf("The layout os tAsB is : {}", tAsB.layout)
        #     cute.print_tensor(tAsB)
        #     cute.print_tensor(tAgB)
        
        cute.copy(thr_copy_mk, tAgA, tAsA)
        cute.copy(thr_copy_nk, tAgB, tAsB)
        cute.arch.cp_async_commit_group()
    ##
    
    ## Main loop
    for k in cutlass.range(nb_tiles_k):
                
        # if tidx==0 and bidx==0 and bidy==0:
        #     cute.printf("##########################################")
        #     cute.printf("Iteration of main loop number %d", k)
        
        if k <= (nb_tiles_k - num_stages):
            
            part_of_tile = (k+num_stages-1)%num_stages
            
            # if tidx==0 and bidx==0 and bidy==0:
            #     cute.printf("The buffer loaded in shared memory is at the location : %d", part_of_tile)
            
            tile_a = ((None, None), (bidx, (k+num_stages-1)))
            tile_b = ((None, None), (bidy, (k+num_stages-1)))
            
            gA = mA[tile_a]
            gB = mB[tile_b]
            
            tAgA = thr_copy_mk.partition_S(gA)
            tAgB = thr_copy_nk.partition_S(gB)
            
            tAsA = thr_copy_mk.partition_D(sA[(None, (None, part_of_tile))])
            tAsB = thr_copy_nk.partition_D(sB[(None, (None, part_of_tile))])
            
            cute.copy(thr_copy_mk, tAgA, tAsA)
            cute.copy(thr_copy_nk, tAgB, tAsB)
            
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(num_stages-1)
        cute.arch.barrier()
        
        # if tidx==0 and bidx==0 and bidy==0:
        #     cute.printf("The gemm done uses the buffers : %d ", k%num_stages)
        
        tile_sA_sB = (None, (None, k%num_stages))
        
        tCsA = thr_mma.partition_A(sA[tile_sA_sB])
        tCsB = thr_mma.partition_B(sB[tile_sA_sB])
        
        tCrA = cute.make_rmem_tensor_like(tCsA, mA.dtype)
        tCrB = cute.make_rmem_tensor_like(tCsB, mB.dtype)
        
        # if tidx==0 and bidx==0 and bidy==0 and k==0:
        #     cute.printf("The layout os tCsA is : {}", tCsA.layout)
        #     cute.printf("The layout of tCrA is : {}", tCrA.layout)
        
        cute.copy(copy_atom_s2r, tCsA, tCrA)
        cute.copy(copy_atom_s2r, tCsB, tCrB)
        
        cute.gemm(tiled_mma, rAcc, tCrA, tCrB, rAcc)
        
        cute.arch.barrier()
    ##
    
    ## Epilogue for storing the result in D
    tCrC.store(rAcc.load().to(cutlass.BFloat16))
    
    gD = mD[tile_cd]
    tCgD = thr_mma.partition_C(gD)
    
    cute.copy(copy_atom_mn, tCrC, tCgD)
    
    


@cute.jit
def _host_kernel_gemm_v3(
    mA : cute.Tensor,
    mB : cute.Tensor,
    mC : cute.Tensor,
    mD : cute.Tensor,
    num_stages : cutlass.Constexpr,
):
    """
    
    """
    _kernel_gemm_v3.set_name_prefix(
        "kernel_gemm_v3",
        remove_cutlass_symbol=True,
        keep_mangled_name=False,
    )
    
    ## Instanciating the tiled_mma
    shape_mnk = (16, 8, 16)
    atom_layout_mnk = (2, 2, 1)
    
    op_mma = cute.nvgpu.warp.MmaF16BF16Op(
        cutlass.BFloat16,
        cutlass.Float32,
        shape_mnk,
    )
    
    permutation_mnk = (
        shape_mnk[0] * atom_layout_mnk[0] * 2,
        shape_mnk[1] * atom_layout_mnk[1] * 4,
        shape_mnk[2] * atom_layout_mnk[2],
    )
    
    tiled_mma = cute.make_tiled_mma(
        op_mma,
        atom_layout_mnk,
        permutation_mnk,
    )
    ##
    
    ## Constraints imposed by tiled mma
    nb_warps = prod(atom_layout_mnk)
    nb_threads = 32 * nb_warps      # 32 threads in each warp
    
    bs_m = cute.size(permutation_mnk[0])
    bs_n = cute.size(permutation_mnk[1])
    bs_k = cute.size(permutation_mnk[2])
    ##
    
    ## Set up for tiled_copy
    nb_elems_tile_mk = bs_m * bs_k
    nb_elems_tile_nk = bs_n * bs_k
    nb_elems_tile_mn = bs_m * bs_n
    
    nb_elems_per_thr_tile_mk = nb_elems_tile_mk // nb_threads
    nb_elems_per_thr_tile_nk = nb_elems_tile_nk // nb_threads
    nb_elems_per_thr_tile_mn = nb_elems_tile_mn // nb_threads
    
    val_layout_mk = cute.make_ordered_layout((1, nb_elems_per_thr_tile_mk), order=(1, 0))
    val_layout_nk = cute.make_ordered_layout((1, nb_elems_per_thr_tile_nk), order=(1, 0))
    val_layout_mn = cute.make_ordered_layout((1, nb_elems_per_thr_tile_mn), order=(1, 0))
    
    thr_layout_mk = cute.make_ordered_layout((bs_m, cute.ceil_div(bs_k, nb_elems_per_thr_tile_mk)), order=(1, 0))
    thr_layout_nk = cute.make_ordered_layout((bs_n ,cute.ceil_div(bs_k, nb_elems_per_thr_tile_nk)), order=(1, 0))
    thr_layout_mn = cute.make_ordered_layout((bs_m, cute.ceil_div(bs_n, nb_elems_per_thr_tile_mn)), order=(1, 0))
    
    tiler_mk, layout_tv_mk = cute.make_layout_tv(
        thr_layout_mk,
        val_layout_mk
    )
    tiler_nk, layout_tv_nk = cute.make_layout_tv(
        thr_layout_nk,
        val_layout_nk,
    )
    tiler_mn, _ = cute.make_layout_tv(
        thr_layout_mn,
        val_layout_mn,
    )
    ##
    
    # print(f"tiler_mn : {tiler_mn} || tiler_mk : {tiler_mk} || tiler_nk : {tiler_nk}")
    
    ## Create copy atoms and tiled copy
    op_atom_async = cute.nvgpu.cpasync.CopyG2SOp()
    op_atom_s2r = cute.nvgpu.CopyS2ROp()
    op_atom_g2r = cute.nvgpu.CopyUniversalOp()
    
    copy_atom_async_mk = cute.make_copy_atom(
        op_atom_async,
        mA.dtype,
        num_bits_per_copy=128,
    )
    copy_atom_async_nk = cute.make_copy_atom(
        op_atom_async,
        mB.dtype,
        num_bits_per_copy=128,
    )

    tiled_copy_async_mk = cute.make_tiled_copy(
        copy_atom_async_mk,
        layout_tv_mk,
        tiler_mk,
    )
    tiled_copy_async_nk = cute.make_tiled_copy(
        copy_atom_async_nk,
        layout_tv_nk,
        tiler_nk,
    )
    
    copy_atom_s2r = cute.make_copy_atom(
        op_atom_s2r,
        mA.dtype,
        num_bits_per_copy=32,
    )
    copy_atom_mn = cute.make_copy_atom(
        op_atom_g2r,
        mC.dtype,
        num_bits_per_copy=16,
    )
    ##
    
    ## Paritioning tensors for kernel
    M = cute.size(mC, mode=[0])
    N = cute.size(mC, mode=[1])
    K = cute.size(mB, mode=[1])
    
    mA = cute.zipped_divide(mA, tiler_mk)
    mB = cute.zipped_divide(mB, tiler_nk)
    mC = cute.zipped_divide(mC, tiler_mn)
    mD = cute.zipped_divide(mD, tiler_mn)
    ##
    
    ## Calculating grid
    grid_m = cute.ceil_div(M, bs_m)
    grid_n = cute.ceil_div(N, bs_n)
    nb_tiles_k = cute.ceil_div(K, bs_k)
    
    args = (
        mA, mB, mC, mD,
        tiled_mma,
        tiled_copy_async_mk,
        tiled_copy_async_nk,
        copy_atom_s2r,
        copy_atom_mn,
        nb_tiles_k,
        bs_m, bs_n, bs_k,
        num_stages,
    )
    
    _kernel_gemm_v3(*args).launch(
        grid=[grid_m, grid_n, 1],
        block=[nb_threads, 1, 1],
    )
    ##
    
    # cA = cute.make_identity_tensor((bs_m, 3*bs_k))
    # cute.printf("The layout of cA is : {}", cA.layout)
    
    # cA = cute.logical_divide(cA, (None, bs_k))
    # cute.printf("The layout of cA after divide is : {}", cA.layout)
    
    # cute.print_tensor(cA[(None, (None, 1))])
    

def gemm_v3(
    a : torch.Tensor,
    b : torch.Tensor,
    c : torch.Tensor,
) -> torch.Tensor:
    """
    Perform a GEMM operation : D = A@B +C.
    
    Parameters
    ----------
    a
        The first input tensor, shape (M, K).
    b
        The second input tensor, shape (K, N).
    c
        The third input tensor, shape (M, N).
    
    Returns
    -------
    d
        The output tensor, shape (M, N).
    
    """
    
    ## Adapting tensor to MMA shapes
    b = b.permute(1, 0).contiguous()
    
    d = torch.empty_like(c)
    ##
    
    ## Adapting tensors to CuTe tensor
    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)
    c_ = from_dlpack(c, assumed_align=16)
    d_ = from_dlpack(d, assumed_align=16)
    ##
    
    num_stages = 4
    
    _host_kernel_gemm_v3(a_, b_, c_, d_, num_stages)
    
    return d
    
    

if __name__ == "__main__":
    M = 128
    N = 128
    K = 128
    
    dtype = torch.bfloat16
    device = torch.device('cuda:0')
    
    torch.manual_seed(42)
    
    a = torch.randn((M, K), dtype=dtype, device=device)
    b = torch.randn((K, N), dtype=dtype, device=device)
    c = torch.randn((M, N), dtype=dtype, device=device)
    
    d = gemm_v3(a, b, c)
    
    d_test = (a.float() @ b.float() + c.float()).to(dtype=dtype)
    
    print(f"The calculated output by the kernel is equal to : \n{d}")
    print(f"The calculated output by PyTorch is equal to : \n{d_test}")
    
    torch.testing.assert_close(d, d_test, atol=1e-2, rtol=1e-2)