import torch
import matplotlib.pyplot as plt

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from gemm_v1 import _kernel_host_gemm_v1


def bench(fn, warmup=100, iters=1000):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters      # ms


def benchmark_gemm_v0():
    """
    Benchmark du GEMM D = A @ B + C (BF16).
    """
    sizes_M = [64, 128, 256, 512, 1024, 2048, 4096]

    dtype = torch.bfloat16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    results = {"size_M": [], "kernel_v0": [], "pytorch": []}

    for M in sizes_M:
        a = torch.randn((M, M), dtype=dtype, device=device)
        b = torch.randn_like(a)
        c = torch.randn_like(a)
        d = torch.empty_like(a)

        a_ = from_dlpack(a, assumed_align=16)
        b_ = from_dlpack(b, assumed_align=16)
        c_ = from_dlpack(c, assumed_align=16)
        d_ = from_dlpack(d, assumed_align=16)

        compiled = cute.compile(_kernel_host_gemm_v1, a_, b_, c_, d_)

        ms_kernel = bench(lambda: compiled(a_, b_, c_, d_))
        ms_torch = bench(lambda: torch.addmm(c, a, b, out=d))

        flops = 2 * M * M * M

        results["size_M"].append(M)
        results["kernel_v0"].append((flops / ms_kernel) * 1e-9)
        results["pytorch"].append((flops / ms_torch) * 1e-9)

    PEAK_TFLOPS = 87.9 * (2.30 / 2.452)     # TFLOP/s

    plt.plot(results["size_M"], results["kernel_v0"], marker="o", label="Custom CuTe DSL Kernel (v1)")
    plt.plot(results["size_M"], results["pytorch"], marker="s", label="PyTorch (cuBLAS)")
    plt.axhline(PEAK_TFLOPS, linestyle="--", color="grey",
                label=f"RTX 5070 Ti Peak throughput ({PEAK_TFLOPS:.1f} TFLOP/s)")

    plt.xscale("log", base=2)
    plt.yscale("log")

    plt.xlabel("Matrix dimension (M)")
    plt.ylabel("Throughput (TFLOP/s)")
    plt.title("GEMM (D = A@B + C) - RTX 5070 Ti, BF16")

    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("benchmarks/figures/benchmark_gemm_v1.png", dpi=600, bbox_inches="tight")


if __name__ == "__main__":
    benchmark_gemm_v0()