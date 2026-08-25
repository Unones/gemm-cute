import torch
from gemm_v2 import gemm_v2

def profile_gemm_v0():
    M = 2048
    N = 2048
    K = 2048
    
    dtype = torch.bfloat16
    device = torch.device("cuda:0")
    
    a = torch.randn((M, K), dtype=dtype, device=device)
    b = torch.randn((K, N), dtype=dtype, device=device)
    c = torch.randn((M, N), dtype=dtype, device=device)
    
    d = gemm_v2(a, b, c)
    
    torch.cuda.synchronize()
    
    torch.cuda.cudart().cudaProfilerStart()
    d = gemm_v2(a, b, c)
    torch.cuda.cudart().cudaProfilerStart()
    
if __name__ == "__main__":
    profile_gemm_v0()