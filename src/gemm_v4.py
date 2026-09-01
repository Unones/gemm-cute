import torch
import cutlass
import cutlass.cute as cute
from math import prod

from cutlass.cute.runtime import from_dlpack


def _make_smem_layout(bs_m, bs_k, num_stages):
    """
    Layout (bs_m, bs_k, num_stages) : (bs_k, 1, bs_m*bs_k), swizzlé.

    Le stage est le mode le plus externe : le slicer ne touche jamais
    les bits manipulés par le swizzle.
    Swizzle<3,3,3> : XOR de m%8 (bits 6-8) dans l'index de chunk 16o (bits 3-5).
    """
    base = cute.make_ordered_layout((bs_m, bs_k, num_stages), order=(1, 0, 2))
    return cute.make_composed_layout(cute.make_swizzle(3, 3, 3), 0, base)


@cute.kernel
def _kernel_gemm_v3(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    mD: cute.Tensor,
    tiled_mma: cute.TiledMma,
    tiled_copy_async_mk: cute.TiledCopy,
    tiled_copy_async_nk: cute.TiledCopy,
    tiled_copy_s2r_mk: cute.TiledCopy,
    tiled_copy_s2r_nk: cute.TiledCopy,
    tiled_copy_s2g_mn: cute.TiledCopy,
    tiled_copy_r2s_mn: cute.TiledCopy,
    copy_atom_g2s_mn: cute.CopyAtom,
    nb_tiles_k: cutlass.Constexpr,
    bs_m: cutlass.Constexpr,
    bs_n: cutlass.Constexpr,
    bs_k: cutlass.Constexpr,
    num_stages: cutlass.Constexpr,
    nb_k_steps: cutlass.Constexpr,
):
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
        _make_smem_layout(bs_m, bs_k, num_stages),
        byte_alignment=128,
    )
    sB = smem.allocate_tensor(
        mB.dtype,
        _make_smem_layout(bs_n, bs_k, num_stages),
        byte_alignment=128,
    )

    ## Chargement de C en registres
    tile_cd = ((None, None), (bidx, bidy))

    gC = mC[tile_cd]
    tCgC = thr_mma.partition_C(gC)
    tCrC = cute.make_rmem_tensor_like(tCgC, mC.dtype)

    cute.copy(copy_atom_g2s_mn, tCgC, tCrC)

    rAcc = cute.make_rmem_tensor_like(tCrC, cutlass.Float32)
    rAcc.store(tCrC.load().to(cutlass.Float32))
    ##

    ## Prologue cp.async
    for i in cutlass.range(num_stages - 1):
        gA = mA[((None, None), (bidx, i))]
        gB = mB[((None, None), (bidy, i))]

        cute.copy(
            thr_copy_g2s_mk,
            thr_copy_g2s_mk.partition_S(gA),
            thr_copy_g2s_mk.partition_D(sA[None, None, i]),
        )
        cute.copy(
            thr_copy_g2s_nk,
            thr_copy_g2s_nk.partition_S(gB),
            thr_copy_g2s_nk.partition_D(sB[None, None, i]),
        )
        cute.arch.cp_async_commit_group()
    ##

    ## Boucle principale
    for k in cutlass.range(nb_tiles_k):

        if k <= (nb_tiles_k - num_stages):
            k_load = k + num_stages - 1
            stage_load = k_load % num_stages

            gA = mA[((None, None), (bidx, k_load))]
            gB = mB[((None, None), (bidy, k_load))]

            cute.copy(
                thr_copy_g2s_mk,
                thr_copy_g2s_mk.partition_S(gA),
                thr_copy_g2s_mk.partition_D(sA[None, None, stage_load]),
            )
            cute.copy(
                thr_copy_g2s_nk,
                thr_copy_g2s_nk.partition_S(gB),
                thr_copy_g2s_nk.partition_D(sB[None, None, stage_load]),
            )

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(num_stages - 1)
        cute.arch.barrier()

        ## Partition UNE SEULE FOIS sur le stage complet.
        ## Le mode k_step apparait comme dernier mode des tenseurs partitionnes,
        ## on l'indexe ensuite sans jamais re-slicer la smem swizzlee.
        stage = k % num_stages
        sA_stage = sA[None, None, stage]
        sB_stage = sB[None, None, stage]

        tCsA = thr_copy_s2r_mk.partition_S(sA_stage)  # (COPY, M, nb_k_steps)
        tCsB = thr_copy_s2r_nk.partition_S(sB_stage)  # (COPY, N, nb_k_steps)

        tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA_stage))
        tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB_stage))

        tCrA_view = thr_copy_s2r_mk.retile(tCrA)
        tCrB_view = thr_copy_s2r_nk.retile(tCrB)

        ## Premier ldmatrix pour masquer la latence
        cute.copy(tiled_copy_s2r_mk, tCsA[None, None, 0], tCrA_view[None, None, 0])
        cute.copy(tiled_copy_s2r_nk, tCsB[None, None, 0], tCrB_view[None, None, 0])
        ##

        for k_step in cutlass.range_constexpr(nb_k_steps):

            if k_step < nb_k_steps - 1:
                cute.copy(
                    tiled_copy_s2r_mk,
                    tCsA[None, None, k_step + 1],
                    tCrA_view[None, None, k_step + 1],
                )
                cute.copy(
                    tiled_copy_s2r_nk,
                    tCsB[None, None, k_step + 1],
                    tCrB_view[None, None, k_step + 1],
                )

            cute.gemm(
                tiled_mma,
                rAcc,
                tCrA[None, None, k_step],
                tCrB[None, None, k_step],
                rAcc,
            )

        cute.arch.barrier()
    ##

    ## Epilogue
    tCrC.store(rAcc.load().to(cutlass.BFloat16))

    ## Meme swizzle sur sD, sinon stmatrix reproduit le conflit 32-way.
    sD = cute.make_tensor(
        sA.iterator,
        cute.make_composed_layout(
            cute.make_swizzle(3, 3, 3),
            0,
            cute.make_ordered_layout((bs_m, bs_n), order=(1, 0)),
        ),
    )

    cute.copy(
        tiled_copy_r2s_mn,
        thr_copy_r2s_mn.retile(tCrC),
        thr_copy_r2s_mn.partition_D(sD),
    )

    cute.arch.barrier()

    gD = mD[tile_cd]
    cute.copy(
        tiled_copy_s2g_mn,
        thr_copy_s2g_mn.partition_S(sD),
        thr_copy_s2g_mn.partition_D(gD),
    )
    ##


@cute.jit
def _host_kernel_gemm_v3(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    mD: cute.Tensor,
    num_stages: cutlass.Constexpr,
):
    _kernel_gemm_v3.set_name_prefix(
        "kernel_gemm_v3",
        remove_cutlass_symbol=True,
        keep_mangled_name=False,
    )

    ## tiled_mma
    shape_mnk = (16, 8, 16)
    atom_layout_mnk = (2, 2, 1)

    op_mma = cute.nvgpu.warp.MmaF16BF16Op(
        cutlass.BFloat16,
        cutlass.Float32,
        shape_mnk,
    )

    permutation_mnk = (
        shape_mnk[0] * atom_layout_mnk[0] * 2,  # 64
        shape_mnk[1] * atom_layout_mnk[1] * 4,  # 64
        shape_mnk[2] * atom_layout_mnk[2],      # 16
    )

    tiled_mma = cute.make_tiled_mma(
        op_mma,
        atom_layout_mnk,
        permutation_mnk,
    )
    ##

    nb_warps = prod(atom_layout_mnk)
    nb_threads = 32 * nb_warps

    bs_m = cute.size(permutation_mnk[0])
    bs_n = cute.size(permutation_mnk[1])
    bs_k = 64

    ## tiled_copy
    nb_elems_per_thr_tile_mk = (bs_m * bs_k) // nb_threads
    nb_elems_per_thr_tile_nk = (bs_n * bs_k) // nb_threads
    nb_elems_per_thr_tile_mn = (bs_m * bs_n) // nb_threads

    val_layout_mk = cute.make_ordered_layout((1, nb_elems_per_thr_tile_mk), order=(1, 0))
    val_layout_nk = cute.make_ordered_layout((1, nb_elems_per_thr_tile_nk), order=(1, 0))
    val_layout_mn = cute.make_ordered_layout((1, nb_elems_per_thr_tile_mn), order=(1, 0))

    thr_layout_mk = cute.make_ordered_layout((bs_m, cute.ceil_div(bs_k, nb_elems_per_thr_tile_mk)), order=(1, 0))
    thr_layout_nk = cute.make_ordered_layout((bs_n, cute.ceil_div(bs_k, nb_elems_per_thr_tile_nk)), order=(1, 0))
    thr_layout_mn = cute.make_ordered_layout((bs_m, cute.ceil_div(bs_n, nb_elems_per_thr_tile_mn)), order=(1, 0))

    tiler_mk, layout_tv_mk = cute.make_layout_tv(thr_layout_mk, val_layout_mk)
    tiler_nk, layout_tv_nk = cute.make_layout_tv(thr_layout_nk, val_layout_nk)
    tiler_mn, layout_tv_mn = cute.make_layout_tv(thr_layout_mn, val_layout_mn)
    ##

    ## copy atoms
    op_atom_async = cute.nvgpu.cpasync.CopyG2SOp()
    op_atom_s2g = cute.nvgpu.CopyUniversalOp()

    copy_atom_async_mk = cute.make_copy_atom(op_atom_async, mA.dtype, num_bits_per_copy=128)
    copy_atom_async_nk = cute.make_copy_atom(op_atom_async, mB.dtype, num_bits_per_copy=128)

    tiled_copy_async_mk = cute.make_tiled_copy(copy_atom_async_mk, layout_tv_mk, tiler_mk)
    tiled_copy_async_nk = cute.make_tiled_copy(copy_atom_async_nk, layout_tv_nk, tiler_nk)

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
    copy_atom_g2s_mn = cute.make_copy_atom(op_atom_s2g, mD.dtype, num_bits_per_copy=16)
    copy_atom_s2g_mn = cute.make_copy_atom(op_atom_s2g, mD.dtype, num_bits_per_copy=128)

    tiled_copy_s2r_mk = cute.make_tiled_copy_A(copy_atom_s2r_mk, tiled_mma)
    tiled_copy_s2r_nk = cute.make_tiled_copy_B(copy_atom_s2r_nk, tiled_mma)
    tiled_copy_r2s_mn = cute.make_tiled_copy_C(copy_atom_r2s_mn, tiled_mma)
    tiled_copy_s2g_mn = cute.make_tiled_copy(copy_atom_s2g_mn, layout_tv_mn, tiler_mn)
    ##

    M = cute.size(mC, mode=[0])
    N = cute.size(mC, mode=[1])
    K = cute.size(mB, mode=[1])

    mA = cute.zipped_divide(mA, tiler_mk)
    mB = cute.zipped_divide(mB, tiler_nk)
    mC = cute.zipped_divide(mC, tiler_mn)
    mD = cute.zipped_divide(mD, tiler_mn)

    grid_m = cute.ceil_div(M, bs_m)
    grid_n = cute.ceil_div(N, bs_n)
    nb_tiles_k = cute.ceil_div(K, bs_k)
    nb_k_steps = cute.ceil_div(bs_k, shape_mnk[2])

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
    )
    _kernel_gemm_v3(*args).launch(
        grid=[grid_m, grid_n, 1],
        block=[nb_threads, 1, 1],
    )


def gemm_v3(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    """
    Perform a GEMM operation : D = A@B.T + C.

    Parameters
    ----------
    a
        The first input tensor, shape (M, K).
    b
        The second input tensor, shape (N, K).
    c
        The third input tensor, shape (M, N).

    Returns
    -------
    d
        The output tensor, shape (M, N).
    """
    d = torch.empty_like(c)

    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)
    c_ = from_dlpack(c, assumed_align=16)
    d_ = from_dlpack(d, assumed_align=16)

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

    torch.testing.assert_close(d, d_test, atol=1e-2, rtol=1e-2)