import torch
import cutlass
import cutlass.cute as cute

from cutlass.cute.runtime import from_dlpack


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


def _get_alignment(input : torch.Tensor, num_bits_per_copy : int = 128):
    """
    Get the maximum number of elements that can be aligned with a single vectorized instruction.
    
    Parameters
    ----------
    input
        The input tensor.
    
    """
    
    num_bytes_per_copy = num_bits_per_copy // 8
    nb_bytes_per_elem = input.element_size()
    nb_elems_per_copy = num_bytes_per_copy // nb_bytes_per_elem
    
    adress = input.data_ptr()
    
    _, N = input.shape
    
    while nb_elems_per_copy > 1:
        if (N % nb_elems_per_copy == 0) and (adress % nb_elems_per_copy == 0):
            break
        nb_elems_per_copy //= 2
    
    return nb_elems_per_copy
    
    


@cute.kernel
def _kernel_gemm_v1(
    mA : cute.Tensor,
    mB : cute.Tensor,
    mC : cute.Tensor,
    mD : cute.Tensor,
    tiled_mma : cute.TiledMma,
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
    
    

@cute.jit
def _jit_gemm_v1(
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
    
    
    
    
def gemm_v1(
    a : torch.Tensor,
    b : torch.Tensor,
    c : torch.Tensor,
):
    """
    Call the kernel _kernel_gemm_v1 to perform the simplest GEMM possible.
    
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
    
    
    ## Set up all the alingments
    nb_elemms_per_copy_a = _get_alignment(a)
    nb_elemms_per_copy_b = _get_alignment(b)
    nb_elemms_per_copy_c = _get_alignment(c)
    
    nb_elems_per_copy_ab = min(nb_elemms_per_copy_a, nb_elemms_per_copy_b)
    alignment_ab = nb_elems_per_copy_ab * a.element_size()
    alignment_c = nb_elemms_per_copy_c * c.element_size()
    ##
    
    a_ = from_dlpack(a, assumed_align=alignment_ab)
    b_ = from_dlpack(b, assumed_align=alignment_ab)
    c_ = from_dlpack(c, assumed_align=alignment_c)
    d_ = from_dlpack(d, assumed_align=alignment_c)
    
    # print(f"The number of bytes aligned of ab is equal to {alignment_ab} bytes.")
    # print(f"The number of bytes aligned of c is equal to {alignment_c} bytes.")
    
    _jit_gemm_v1(a_, b_, c_, d_)
    
    
    
    
if __name__ == "__main__":
    M = 4
    N = 4
    K = 8
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    a = torch.randn((M, K), dtype=dtype, device=device)
    b = torch.randn((K, N), dtype=dtype, device=device)
    c = torch.randn((M, N), dtype=dtype, device=device)
    
    gemm_v1(a, b, c)