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
    tiled_copy_mk : cute.TiledCopy,
    tiled_copy_nk : cute.TiledCopy,
    tiled_copy_mn : cute.TiledCopy,
    nb_tiled_k : cutlass.Constexpr,
):
    """
    
    """
    
    
    

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
    
    nb_elems_per_thr = 8    # vectorization bf16
    
    val_layout = cute.make_layout((1, nb_elems_per_thr), stride=(0, 1))
    thr_layout_mk = cute.make_layout((bs_m, bs_k//nb_elems_per_thr), stride=(bs_k//nb_elems_per_thr, 1))
    thr_layout_nk = cute.make_layout((bs_n, bs_k // nb_elems_per_thr), stride=(bs_k // nb_elems_per_thr, 1))
    thr_layout_mn = cute.make_layout((bs_m, bs_n // nb_elems_per_thr), stride=(bs_n // nb_elems_per_thr, 1))
    
    op_copy = cute.nvgpu.CopyUniversalOp()
    copy_atom = cute.make_copy_atom(
        op_copy,
        mA.dtype,
        num_bites_per_copy=128,
    )
    
    tiler_mk, layout_tv_mk = cute.make_layout_tv(
        thr_layout_mk,
        val_layout,
    )
    tiler_nk, layout_tv_nk = cute.make_layout_tv(
        thr_layout_nk,
        val_layout,
    )
    tiler_mn, layout_tv_mn = cute.make_layout_tv(
        thr_layout_mn,
        val_layout,
    )
    
    tiled_copy_mk = cute.make_tiled_copy(
        copy_atom,
        layout_tv_mk,
        tiler_mk,
    )
    tiled_copy_nk = cute.make_tiled_copy(
        copy_atom,
        layout_tv_nk,
        tiler_nk,
    )
    tiled_copy_mn = cute.make_tiled_copy(
        copy_atom,
        layout_tv_mn,
        tiler_mn,
    )
    ##
    
    ## Set up the tensors
    mA = cute.zipped_divide(mA, tiler_mk)
    mB = cute.zipped_divide(mB, tiler_nk)
    mC = cute.zipped_divide(mC, tiler_mn)
    mD = cute.zipped_divide(mD, tiler_mn)
    ##
    
    ## Initializing the grid
    nb_warps_per_block = prod(atom_layout_mnk)
    nb_theads_per_block = 32 * nb_warps_per_block   # 32 threads per warp
    
    grid_m = cute.ceil_div(M, bs_m)
    grid_n = cute.ceil_div(N, bs_n)
    nb_tiled_k = cute.ceil_div(K, bs_k)
    ##
    
    ## Launching the kernel
    args = (
        mA, mB, mC, mD,
        tiled_mma,
        tiled_copy_mk,
        tiled_copy_nk,
        tiled_copy_mn,
        nb_tiled_k,
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
    M = 32
    N = 32
    K = 32
    
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
    
    # torch.testing.assert_close(d, d_test, atol=1e-2, rtol=1e-2)