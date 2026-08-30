import torch
import pytest

from gemm_v3 import gemm_v3

@pytest.mark.parametrize("M", [256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("N", [256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("K", [256, 512, 1024, 2048, 4096])
def test_gemm_v2(M, N, K):
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    torch.manual_seed(42)
    
    a = torch.randn((M, K), dtype=dtype, device=device)
    b = torch.randn((N, K), dtype=dtype, device=device)
    c = torch.randn((M, N), dtype=dtype, device=device)
    
    d = gemm_v3(a, b, c)
    
    b = b.T
    d_test = (a.float()@b.float() +c.float()).to(dtype=dtype)
    
    torch.testing.assert_close(d, d_test, atol=1e-2, rtol=1e-2)