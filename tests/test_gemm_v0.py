import torch
import pytest

from gemm_v0 import gemm_v0

@pytest.mark.parametrize("M", [256, 512, 1024, 2048])
@pytest.mark.parametrize("N", [256, 512, 1024, 2048])
@pytest.mark.parametrize("K", [256, 512, 1024, 2048])
def test_gemm_v0(M, N, K):
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    a = torch.randn((M, K), dtype=dtype, device=device)
    b = torch.randn((K, N), dtype=dtype, device=device)
    c = torch.randn((M, N), dtype=dtype, device=device)
    
    d = gemm_v0(a, b, c)
    
    d_test = (a.float()@b.float() +c.float()).to(dtype=dtype)
    
    # print(f"The output calculated by the kernel is equal to : \n{d}")
    # print(f"The output calculated by PyTorch is equal to : \n{d_test}")
    
    torch.testing.assert_close(d, d_test, atol=1e-2, rtol=1e-2)