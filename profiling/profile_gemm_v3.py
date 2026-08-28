import torch
from gemm_v3 import gemm_v3

def profile_gemm_v3():
    M = 3048
    N = 3048
    K = 3048
    
    dtype = torch.bfloat16
    device = torch.device("cuda:0")
    
    a = torch.randn((M, K), dtype=dtype, device=device)
    b = torch.randn((K, N), dtype=dtype, device=device)
    c = torch.randn((M, N), dtype=dtype, device=device)
    
    d = gemm_v3(a, b, c)
    
    torch.cuda.synchronize()
    
    torch.cuda.cudart().cudaProfilerStart()
    d = gemm_v3(a, b, c)
    torch.cuda.cudart().cudaProfilerStart()
    
if __name__ == "__main__":
    profile_gemm_v3()