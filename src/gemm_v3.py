import torch
import cutlass
import cutlass.cute as cute
from math import prod

from cutlass.cute.runtime import from_dlpack


@cute.jit
def _flat_stage(s : cute.Tensor, i : cutlass.Numeric):
    s_grouped = cute.group_modes(s[(None, (None, None, i))], 1, 3)
    s_coalesced = cute.coalesce(s_grouped, target_profile=(1, 1))
    return s_coalesced



@cute.kernel
def _kernel_gemm_v3(
    mA : cute.Tensor,
    mB : cute.Tensor,
    mC : cute.Tensor,
    mD : cute.Tensor,
    tiled_mma : cute.TiledMma,
    tiled_copy_async_mk : cute.TiledCopy,
    tiled_copy_async_nk : cute.TiledCopy,
    tiled_copy_s2r_mk : cute.TiledCopy,
    tiled_copy_s2r_nk : cute.TiledCopy,
    tiled_copy_s2g_mn : cute.TiledCopy,
    tiled_copy_r2s_mn : cute.TiledCopy,
    copy_atom_g2s_mn : cute.CopyAtom,
    nb_tiles_k : cutlass.Constexpr,
    bs_m : cutlass.Constexpr,
    bs_n : cutlass.Constexpr,
    bs_k : cutlass.Constexpr,
    num_stages : cutlass.Constexpr,
    nb_k_steps : cutlass.Constexpr,
    size_atom_k : cutlass.Constexpr,
    
):
    """
    
    
    """
    
    ## Initiate needed variables
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    
    thr_copy_g2s_mk = tiled_copy_async_mk.get_slice(tidx)
    thr_copy_g2s_nk = tiled_copy_async_nk.get_slice(tidx)
    thr_copy_s2r_mk = tiled_copy_s2r_mk.get_slice(tidx)
    thr_copy_s2r_nk = tiled_copy_s2r_nk.get_slice(tidx)
    thr_copy_r2s_mn = tiled_copy_r2s_mn.get_slice(tidx)
    thr_copy_s2g_mn = tiled_copy_s2g_mn.get_slice(tidx)
    thr_mma = tiled_mma.get_slice(tidx)
    
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(
        mA.dtype,
        cute.make_ordered_layout((bs_m, (size_atom_k, nb_k_steps, num_stages)), order=(2, (0, 1, 3))),
        byte_alignment=128,
    )
    sB = smem.allocate_tensor(
        mB.dtype,
        cute.make_ordered_layout((bs_n, (size_atom_k, nb_k_steps, num_stages)), order=(2, (0, 1, 3))),
        byte_alignment=128,
    )
    ##
    
    ## Loading C elements to reegisters
    tile_cd = ((None, None), (bidx, bidy))
    
    gC = mC[tile_cd]
    tCgC = thr_mma.partition_C(gC)
    tCrC = cute.make_rmem_tensor_like(tCgC, mC.dtype)
    
    cute.copy(copy_atom_g2s_mn, tCgC, tCrC)
    
    rAcc = cute.make_rmem_tensor_like(tCrC, cutlass.Float32)
    rAcc.fill(0.0)
    rAcc.store(tCrC.load().to(cutlass.Float32))
    ##
    
    ## Prologue for async copy
    for i in cutlass.range(num_stages - 1):
        
        tile_a = ((None, None), (bidx, i))
        tile_b = ((None, None), (bidy, i))
        
        gA = mA[tile_a]
        gB = mB[tile_b]
        
        tAgA = thr_copy_g2s_mk.partition_S(gA)
        tAgB = thr_copy_g2s_nk.partition_S(gB)
        
        tAsA = thr_copy_g2s_mk.partition_D(_flat_stage(sA, i))
        tAsB = thr_copy_g2s_nk.partition_D(_flat_stage(sB, i))
        
        cute.copy(thr_copy_g2s_mk, tAgA, tAsA)
        cute.copy(thr_copy_g2s_nk, tAgB, tAsB)
        cute.arch.cp_async_commit_group()
    ##
    
    ## Main loop
    for k in cutlass.range(nb_tiles_k):
        
        ## Async copy if not out of bound of tile
        if k <= (nb_tiles_k - num_stages):
            
            part_of_tile = (k+num_stages-1)%num_stages
            
            tile_a = ((None, None), (bidx, (k+num_stages-1)))
            tile_b = ((None, None), (bidy, (k+num_stages-1)))
            
            gA = mA[tile_a]
            gB = mB[tile_b]
            
            tAgA = thr_copy_g2s_mk.partition_S(gA)
            tAgB = thr_copy_g2s_nk.partition_S(gB)
            
            tAsA = thr_copy_g2s_mk.partition_D(_flat_stage(sA, part_of_tile))
            tAsB = thr_copy_g2s_nk.partition_D(_flat_stage(sB, part_of_tile))
            
            cute.copy(thr_copy_g2s_mk, tAgA, tAsA)
            cute.copy(thr_copy_g2s_nk, tAgB, tAsB)
        ##
        
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(num_stages-1)
        cute.arch.barrier()
        
        ## First ldmatrix for hiding latency
        buffer_part = k%num_stages
        first_part_tile_sA_sB = (None, (None, 0, buffer_part))
        
        tCsA = thr_copy_s2r_mk.partition_S(sA[first_part_tile_sA_sB])
        tCsB = thr_copy_s2r_nk.partition_S(sB[first_part_tile_sA_sB])
        
        tCrA_actual = tiled_mma.make_fragment_A(thr_mma.partition_A(sA[first_part_tile_sA_sB]))
        tCrB_actual = tiled_mma.make_fragment_B(thr_mma.partition_B(sB[first_part_tile_sA_sB]))
        
        cute.copy(tiled_copy_s2r_mk, tCsA, thr_copy_s2r_mk.retile(tCrA_actual))
        cute.copy(tiled_copy_s2r_nk, tCsB, thr_copy_s2r_nk.retile(tCrB_actual))
        ##
        
        ## Inner Loop to load parts of buffer
        for k_step in cutlass.range_constexpr(nb_k_steps):
            
            tCrA = tCrA_actual
            tCrB = tCrB_actual
            
            if k_step < nb_k_steps -1 :
                tile_sA_sB = (None, (None, k_step+1, buffer_part))
                tCsA = thr_copy_s2r_mk.partition_S(sA[tile_sA_sB])
                tCsB = thr_copy_s2r_nk.partition_S(sB[tile_sA_sB])
                
                tCrA_next = tiled_mma.make_fragment_A(thr_mma.partition_A(sA[tile_sA_sB]))
                tCrB_next = tiled_mma.make_fragment_B(thr_mma.partition_B(sB[tile_sA_sB]))
                
                cute.copy(tiled_copy_s2r_mk, tCsA, thr_copy_s2r_mk.retile(tCrA_next))
                cute.copy(tiled_copy_s2r_nk, tCsB, thr_copy_s2r_nk.retile(tCrB_next))   # problem with compute-sanitizer
                
                tCrA_actual = tCrA_next
                tCrB_actual = tCrB_next

            cute.gemm(tiled_mma, rAcc, tCrA, tCrB, rAcc)
            
        cute.arch.barrier()
        ##
    
    ## Epilogue for storing the result in D
    tCrC.store(rAcc.load().to(cutlass.BFloat16))
    
    sD = cute.make_tensor(
        sA.iterator,
        cute.make_ordered_layout((bs_m, bs_n), order=(1, 0)),
    )
    tCsD = thr_copy_r2s_mn.partition_D(sD)
    
    cute.copy(tiled_copy_r2s_mn, thr_copy_r2s_mn.retile(tCrC), tCsD)
    
    cute.arch.barrier()
    
    gD = mD[tile_cd]
    tAgD = thr_copy_s2g_mn.partition_D(gD)
    tAsD = thr_copy_s2g_mn.partition_S(sD)
    
    cute.copy(tiled_copy_s2g_mn, tAsD, tAgD)
    # ##
    
    


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
        shape_mnk[0] * atom_layout_mnk[0] * 4,  # 128
        shape_mnk[1] * atom_layout_mnk[1] * 8,  # 128
        shape_mnk[2] * atom_layout_mnk[2],      # 16
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
    bs_k = 64
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
    tiler_mn, layout_tv_mn = cute.make_layout_tv(
        thr_layout_mn,
        val_layout_mn,
    )
    ##
    
    # print(f"tiler_mn : {tiler_mn} || tiler_mk : {tiler_mk} || tiler_nk : {tiler_nk}")
    # print(f"bs_m : {bs_m} || bs_n : {bs_n} || bs_k : {bs_k}")
    
    ## Create copy atoms and tiled copy
    op_atom_async = cute.nvgpu.cpasync.CopyG2SOp()
    op_atom_s2g = cute.nvgpu.CopyUniversalOp()
    
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
    
    copy_atom_s2r_mk = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
        mA.dtype,
    )
    copy_atom_s2r_nk = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
        mB.dtype,
    )
    copy_atom_r2s_mn = cute.make_copy_atom(
        cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4),
        mD.dtype,
    )
    copy_atom_g2s_mn = cute.make_copy_atom(
        op_atom_s2g,
        mD.dtype,
        num_bits_per_copy=16,
    )
    copy_atom_s2g_mn = cute.make_copy_atom(
        op_atom_s2g,
        mD.dtype,
        num_bits_per_copy=128,
    )
    
    tiled_copy_s2r_mk = cute.make_tiled_copy_A(
        copy_atom_s2r_mk,
        tiled_mma,
    )
    tiled_copy_s2r_nk = cute.make_tiled_copy_B(
        copy_atom_s2r_nk,
        tiled_mma,
    )
    tiled_copy_r2s_mn = cute.make_tiled_copy_C(
        copy_atom_r2s_mn,
        tiled_mma,
    )
    tiled_copy_s2g_mn = cute.make_tiled_copy(
        copy_atom_s2g_mn,
        layout_tv_mn,
        tiler_mn,
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
    nb_k_steps = cute.ceil_div(bs_k, shape_mnk[2])
    size_atom_k = shape_mnk[2]
    
    args = (
        mA, mB, mC, mD,
        tiled_mma,
        tiled_copy_async_mk,
        tiled_copy_async_nk,
        tiled_copy_s2r_mk,
        tiled_copy_s2r_nk,
        tiled_copy_s2g_mn,
        tiled_copy_r2s_mn,
        copy_atom_g2s_mn,
        nb_tiles_k,
        bs_m, bs_n, bs_k,
        num_stages,
        nb_k_steps,
        size_atom_k,
    )
    _kernel_gemm_v3(*args).launch(
        grid=[grid_m, grid_n, 1],
        block=[nb_threads, 1, 1],
    )
    
    ##
    

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
    
    d = torch.empty_like(c)
    ##
    
    ## Adapting tensors to CuTe tensor
    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)
    c_ = from_dlpack(c, assumed_align=16)
    d_ = from_dlpack(d, assumed_align=16)
    ##
    
    num_stages = 2
    
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
    b = torch.randn((N, K), dtype=dtype, device=device)
    c = torch.randn((M, N), dtype=dtype, device=device)
    
    d = gemm_v3(a, b, c)
    
    b = b.T
    d_test = (a.float() @ b.float() + c.float()).to(dtype=dtype)
    
    # print(f"The calculated output by the kernel is equal to : \n{d}")
    # print(f"The calculated output by PyTorch is equal to : \n{d_test}")
    
    torch.testing.assert_close(d, d_test, atol=1e-2, rtol=1e-2)