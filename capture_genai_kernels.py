"""
Capture PyTorch GenAI benchmark kernels at ALL their defined shapes.

Source: /tmp/pytorch-work/benchmarks/dynamo/genai_layers/kernels.py

8 kernel classes:
- CrossEntropyForward/Backward (11 shapes each)
- SoftmaxForward/Backward (11 shapes each)
- RMSNormForward/Backward (15/11 shapes each, includes extra_shapes_for_norm)
- LayerNormForward/Backward (13 shapes each, includes extra_shapes_for_norm)
"""
import os
import sys
import tempfile

sys.path.insert(0, "/tmp/scratch_space/better_benchmark")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn.functional as F
from pathlib import Path

from capture_hook import install_capture_hook, uninstall_capture_hook
from merge_captures import merge_one_capture
from torch._inductor.utils import fresh_inductor_cache

REPRO_DIR = Path("/tmp/scratch_space/better_benchmark/repros")
DEVICE = "cuda"
DTYPE = torch.bfloat16

# Shape definitions from kernels.py
extra_shapes_for_norm = (
    (1152 * 500, 384),    # (576000, 384)
    (1152 * 500, 512),    # (576000, 512)
    (1152 * 1000, 384),   # (1152000, 384)
    (1152 * 1000, 512),   # (1152000, 512)
)

CROSS_ENTROPY_SHAPES = (
    (32768, 256),
    (32768, 512),
    (32768, 1024),
    (32768, 2048),
    (32768, 4096),
    (32768, 8192),
    (32768, 16384),
    (32768, 32768),
    (32768, 65536),
    (16384, 131072),
    (8192, 262144),
)

SOFTMAX_SHAPES = (
    (32768, 256),
    (32768, 512),
    (32768, 1024),
    (32768, 2048),
    (32768, 4096),
    (32768, 8192),
    (32768, 16384),
    (32768, 32768),
    (32768, 65536),
    (16384, 131072),
    (8192, 262144),
)

RMSNORM_FWD_SHAPES = (
    (32768, 256),
    (32768, 512),
    (32768, 1024),
    (32768, 2048),
    (32768, 4096),
    (32768, 8192),
    (32768, 16384),
    (32768, 32768),
    (32768, 65536),
    (16384, 131072),
    (8192, 262144),
) + extra_shapes_for_norm

RMSNORM_BWD_SHAPES = (
    (32768, 256),
    (32768, 512),
    (32768, 1024),
    (32768, 2048),
    (32768, 4096),
    (32768, 8192),
    (32768, 16384),
) + extra_shapes_for_norm

LAYERNORM_FWD_SHAPES = (
    (32768, 256),
    (32768, 512),
    (32768, 1024),
    (32768, 2048),
    (32768, 4096),
    (32768, 8192),
    (32768, 16384),
    (32768, 32768),
    (32768, 65536),
) + extra_shapes_for_norm

LAYERNORM_BWD_SHAPES = (
    (32768, 256),
    (32768, 512),
    (32768, 1024),
    (32768, 2048),
    (32768, 4096),
    (32768, 8192),
    (32768, 16384),
    (32768, 32768),
    (32768, 65536),
) + extra_shapes_for_norm


def capture_forward(fn, label, tmpdir):
    """Capture a forward-only computation."""
    torch._dynamo.reset()
    install_capture_hook(tmpdir, label=label)
    try:
        compiled = torch.compile(fn, fullgraph=True)
        with fresh_inductor_cache():
            with torch.no_grad():
                compiled()
                torch.cuda.synchronize()
    except Exception as e:
        print(f"  ERROR capturing {label}: {e}")
    uninstall_capture_hook()


def capture_backward(fwd_fn, label, tmpdir):
    """Capture a backward computation via autograd."""
    torch._dynamo.reset()
    install_capture_hook(tmpdir, label=label)
    try:
        compiled = torch.compile(fwd_fn, fullgraph=True)
        with fresh_inductor_cache():
            with torch.enable_grad():
                loss = compiled()
                loss.backward()
                torch.cuda.synchronize()
    except Exception as e:
        print(f"  ERROR capturing {label}: {e}")
    uninstall_capture_hook()


def run_all():
    results = {}

    # ======================================================================
    # CrossEntropyForward
    # ======================================================================
    print("\n" + "=" * 70)
    print("CrossEntropyForward - 11 shapes")
    print("=" * 70)
    count = 0
    for M, N in CROSS_ENTROPY_SHAPES:
        label = f"genai_ce_fwd_{M}x{N}"
        print(f"  [{M}, {N}]...", end=" ", flush=True)
        tmpdir = tempfile.mkdtemp()
        x = torch.randn(M, N, device=DEVICE, dtype=DTYPE)
        target = torch.randint(0, N, (M,), device=DEVICE)

        def ce_fwd(x=x, target=target):
            return F.cross_entropy(x, target, reduction="none")

        capture_forward(ce_fwd, label, tmpdir)
        n = merge_one_capture(Path(tmpdir), REPRO_DIR, label)
        count += n
        print(f"merged {n}")
        del x, target
        torch.cuda.empty_cache()
    results["CrossEntropyForward"] = count
    print(f"  Total: {count} regions")

    # ======================================================================
    # CrossEntropyBackward
    # ======================================================================
    print("\n" + "=" * 70)
    print("CrossEntropyBackward - 11 shapes")
    print("=" * 70)
    count = 0
    for M, N in CROSS_ENTROPY_SHAPES:
        label = f"genai_ce_bwd_{M}x{N}"
        print(f"  [{M}, {N}]...", end=" ", flush=True)
        tmpdir = tempfile.mkdtemp()
        x = torch.randn(M, N, device=DEVICE, dtype=DTYPE, requires_grad=True)
        target = torch.randint(0, N, (M,), device=DEVICE)

        def ce_bwd(x=x, target=target):
            loss = F.cross_entropy(x, target, reduction="none")
            return loss.sum()

        capture_backward(ce_bwd, label, tmpdir)
        n = merge_one_capture(Path(tmpdir), REPRO_DIR, label)
        count += n
        print(f"merged {n}")
        del x, target
        torch.cuda.empty_cache()
    results["CrossEntropyBackward"] = count
    print(f"  Total: {count} regions")

    # ======================================================================
    # SoftmaxForward
    # ======================================================================
    print("\n" + "=" * 70)
    print("SoftmaxForward - 11 shapes")
    print("=" * 70)
    count = 0
    for M, N in SOFTMAX_SHAPES:
        label = f"genai_softmax_fwd_{M}x{N}"
        print(f"  [{M}, {N}]...", end=" ", flush=True)
        tmpdir = tempfile.mkdtemp()
        x = torch.randn(M, N, device=DEVICE, dtype=DTYPE)

        def sm_fwd(x=x):
            return F.softmax(x, dim=-1)

        capture_forward(sm_fwd, label, tmpdir)
        n = merge_one_capture(Path(tmpdir), REPRO_DIR, label)
        count += n
        print(f"merged {n}")
        del x
        torch.cuda.empty_cache()
    results["SoftmaxForward"] = count
    print(f"  Total: {count} regions")

    # ======================================================================
    # SoftmaxBackward
    # ======================================================================
    print("\n" + "=" * 70)
    print("SoftmaxBackward - 11 shapes")
    print("=" * 70)
    count = 0
    for M, N in SOFTMAX_SHAPES:
        label = f"genai_softmax_bwd_{M}x{N}"
        print(f"  [{M}, {N}]...", end=" ", flush=True)
        tmpdir = tempfile.mkdtemp()
        x = torch.randn(M, N, device=DEVICE, dtype=DTYPE, requires_grad=True)

        def sm_bwd(x=x):
            y = F.softmax(x, dim=-1)
            return y.sum()

        capture_backward(sm_bwd, label, tmpdir)
        n = merge_one_capture(Path(tmpdir), REPRO_DIR, label)
        count += n
        print(f"merged {n}")
        del x
        torch.cuda.empty_cache()
    results["SoftmaxBackward"] = count
    print(f"  Total: {count} regions")

    # ======================================================================
    # RMSNormForward
    # ======================================================================
    print("\n" + "=" * 70)
    print(f"RMSNormForward - {len(RMSNORM_FWD_SHAPES)} shapes")
    print("=" * 70)
    count = 0
    for M, N in RMSNORM_FWD_SHAPES:
        label = f"genai_rmsnorm_fwd_{M}x{N}"
        print(f"  [{M}, {N}]...", end=" ", flush=True)
        tmpdir = tempfile.mkdtemp()
        x = torch.randn(M, N, device=DEVICE, dtype=DTYPE)
        w = torch.randn(N, device=DEVICE, dtype=torch.float32)

        def rmsnorm_fwd(x=x, w=w):
            x_f32 = x.float()
            return (
                x_f32
                * torch.rsqrt(torch.mean(x_f32.square(), dim=-1, keepdim=True) + 1e-6)
                * w
            ).to(x.dtype)

        capture_forward(rmsnorm_fwd, label, tmpdir)
        n = merge_one_capture(Path(tmpdir), REPRO_DIR, label)
        count += n
        print(f"merged {n}")
        del x, w
        torch.cuda.empty_cache()
    results["RMSNormForward"] = count
    print(f"  Total: {count} regions")

    # ======================================================================
    # RMSNormBackward
    # ======================================================================
    print("\n" + "=" * 70)
    print(f"RMSNormBackward - {len(RMSNORM_BWD_SHAPES)} shapes")
    print("=" * 70)
    count = 0
    for M, N in RMSNORM_BWD_SHAPES:
        label = f"genai_rmsnorm_bwd_{M}x{N}"
        print(f"  [{M}, {N}]...", end=" ", flush=True)
        tmpdir = tempfile.mkdtemp()
        x = torch.randn(M, N, device=DEVICE, dtype=DTYPE, requires_grad=True)
        w = torch.randn(N, device=DEVICE, dtype=torch.float32, requires_grad=True)

        def rmsnorm_bwd(x=x, w=w):
            x_f32 = x.float()
            out = (
                x_f32
                * torch.rsqrt(torch.mean(x_f32.square(), dim=-1, keepdim=True) + 1e-6)
                * w
            ).to(x.dtype)
            return out.sum()

        capture_backward(rmsnorm_bwd, label, tmpdir)
        n = merge_one_capture(Path(tmpdir), REPRO_DIR, label)
        count += n
        print(f"merged {n}")
        del x, w
        torch.cuda.empty_cache()
    results["RMSNormBackward"] = count
    print(f"  Total: {count} regions")

    # ======================================================================
    # LayerNormForward
    # ======================================================================
    print("\n" + "=" * 70)
    print(f"LayerNormForward - {len(LAYERNORM_FWD_SHAPES)} shapes")
    print("=" * 70)
    count = 0
    for M, N in LAYERNORM_FWD_SHAPES:
        label = f"genai_layernorm_fwd_{M}x{N}"
        print(f"  [{M}, {N}]...", end=" ", flush=True)
        tmpdir = tempfile.mkdtemp()
        x = torch.randn(M, N, device=DEVICE, dtype=DTYPE)
        w = torch.randn(N, device=DEVICE, dtype=torch.float32)

        def layernorm_fwd(x=x, w=w):
            x_f32 = x.float()
            return F.layer_norm(x_f32, w.shape, w, None, 1e-6).to(x.dtype)

        capture_forward(layernorm_fwd, label, tmpdir)
        n = merge_one_capture(Path(tmpdir), REPRO_DIR, label)
        count += n
        print(f"merged {n}")
        del x, w
        torch.cuda.empty_cache()
    results["LayerNormForward"] = count
    print(f"  Total: {count} regions")

    # ======================================================================
    # LayerNormBackward
    # ======================================================================
    print("\n" + "=" * 70)
    print(f"LayerNormBackward - {len(LAYERNORM_BWD_SHAPES)} shapes")
    print("=" * 70)
    count = 0
    for M, N in LAYERNORM_BWD_SHAPES:
        label = f"genai_layernorm_bwd_{M}x{N}"
        print(f"  [{M}, {N}]...", end=" ", flush=True)
        tmpdir = tempfile.mkdtemp()
        x = torch.randn(M, N, device=DEVICE, dtype=DTYPE, requires_grad=True)
        w = torch.randn(N, device=DEVICE, dtype=torch.float32, requires_grad=True)

        def layernorm_bwd(x=x, w=w):
            x_f32 = x.float()
            out = F.layer_norm(x_f32, w.shape, w, None, 1e-6).to(x.dtype)
            return out.sum()

        capture_backward(layernorm_bwd, label, tmpdir)
        n = merge_one_capture(Path(tmpdir), REPRO_DIR, label)
        count += n
        print(f"merged {n}")
        del x, w
        torch.cuda.empty_cache()
    results["LayerNormBackward"] = count
    print(f"  Total: {count} regions")

    # ======================================================================
    # Summary
    # ======================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total = 0
    for kernel, cnt in results.items():
        print(f"  {kernel}: {cnt} regions merged")
        total += cnt
    print(f"\n  TOTAL: {total} regions merged across all kernels")
    return results


if __name__ == "__main__":
    run_all()
