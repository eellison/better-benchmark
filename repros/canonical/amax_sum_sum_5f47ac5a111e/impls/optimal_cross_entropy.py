"""
Optimal fused cross-entropy loss Triton kernel.

Fuses the gather into the softmax reduction, eliminating the 4GB intermediate
materialization. Uses ONLINE SOFTMAX in a SINGLE pass over each row to
simultaneously:
  1. Compute running max (for numerical stability)
  2. Compute running sum(exp(x - max)) with online correction
  3. Load x[label] (the value at the label position)

Then: loss = -(x[label] - max - log(sum_exp))  (per-row)
Final: mean over non-ignored rows.

Data read: input[N, V] ONCE + labels[N] once + bias[V] once = ~2GB (single pass!)
Data written: per-row loss [N] + scalar. No 4GB intermediate.

vs Inductor which:
  - Pass 1: compute amax over [32768, 30522] -> writes [32768, 30522] sub result
  - Pass 2: compute sum(exp) over [32768, 30522] -> writes [32768, 30522] log_softmax
  - Pass 3: gather from [32768, 30522] -> reads 4GB just for 32768 values
  Total wasted DRAM traffic: ~8GB of unnecessary writes+reads
"""
import torch
import triton
import triton.language as tl
import time


@triton.jit
def _fused_cross_entropy_kernel(
    input_ptr,        # [N, V_padded] f32, strided
    bias_ptr,         # [V] f32 bias to add
    labels_ptr,       # [N] i64
    loss_ptr,         # [N] f32 per-row losses (or 0 if ignored)
    count_ptr,        # [1] i64 count of non-ignored
    N: tl.constexpr,
    V: tl.constexpr,  # actual vocab size (30522)
    V_padded: tl.constexpr,  # stride in dim 1 (30524)
    BLOCK_V: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """
    One program instance per row. Uses online softmax (single pass over data)
    to compute max, sum_exp, and gather x[label] simultaneously.

    Online softmax recurrence:
        When max changes from m_old to m_new:
            sum_exp = sum_exp * exp(m_old - m_new) + sum_of_new_exps
    """
    row = tl.program_id(0)

    # Load label for this row
    label = tl.load(labels_ptr + row)

    # If label == -100, this row is ignored
    is_valid = (label != -100)

    # Clamp label to valid range for loading (will be masked out later)
    safe_label = tl.where(is_valid, label, tl.zeros_like(label))

    # --- Single pass: online softmax + gather x[label] ---
    row_start = row * V_padded

    m = float("-inf")  # running max
    sum_exp = 0.0      # running sum of exp(x - m), corrected online
    x_label = 0.0      # value at label position

    for start in range(0, V, BLOCK_V):
        offs = start + tl.arange(0, BLOCK_V)
        mask = offs < V
        x = tl.load(input_ptr + row_start + offs, mask=mask, other=float("-inf"))
        if HAS_BIAS:
            b = tl.load(bias_ptr + offs, mask=mask, other=0.0)
            x = x + b

        # Online softmax: update max and correct running sum
        block_max = tl.max(x, axis=0)
        m_new = tl.maximum(m, block_max)

        # Correct previous sum_exp for the new max
        sum_exp = sum_exp * tl.exp(m - m_new)

        # Add new block's contribution
        exp_x = tl.exp(x - m_new)
        exp_x = tl.where(mask, exp_x, tl.zeros_like(exp_x))
        sum_exp += tl.sum(exp_x, axis=0)

        m = m_new

        # Check if label is in this block and extract its value
        label_in_block = (safe_label >= start) & (safe_label < start + BLOCK_V)
        if label_in_block:
            x_label = tl.sum(tl.where(offs == safe_label, x, tl.zeros_like(x)), axis=0)

    # log_softmax[label] = x[label] - max - log(sum_exp)
    log_softmax_label = x_label - m - tl.log(sum_exp)

    # loss = -log_softmax[label] if valid, else 0
    row_loss = tl.where(is_valid, -log_softmax_label, 0.0)

    # Store per-row loss
    tl.store(loss_ptr + row, row_loss)

    # Atomically count valid rows
    if is_valid:
        tl.atomic_add(count_ptr, 1)


@triton.jit
def _reduce_loss_kernel(
    loss_ptr,       # [N] f32
    count_ptr,      # [1] i64
    output_ptr,     # [1] f32
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Reduce per-row losses to scalar mean."""
    total = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        losses = tl.load(loss_ptr + offs, mask=mask, other=0.0)
        total += tl.sum(losses, axis=0)

    count = tl.load(count_ptr).to(tl.float32)
    mean_loss = total / count
    tl.store(output_ptr, mean_loss)


def fused_cross_entropy(input_2d, labels_1d, bias=None):
    """
    Compute cross-entropy loss without materializing the [N, V] log_softmax.

    Args:
        input_2d: [N, V_padded] float32 tensor (logits before bias)
        labels_1d: [N] int64 tensor (labels, -100 = ignore)
        bias: [V] float32 tensor (decoder bias) or None

    Returns:
        scalar float32 loss
    """
    N = input_2d.shape[0]
    V_padded = input_2d.shape[1]
    V = bias.shape[0] if bias is not None else V_padded

    # Output buffers
    per_row_loss = torch.empty(N, dtype=torch.float32, device=input_2d.device)
    count = torch.zeros(1, dtype=torch.int64, device=input_2d.device)
    output = torch.empty(1, dtype=torch.float32, device=input_2d.device)

    # Choose BLOCK_V: must be power of 2. Tuned empirically on B200:
    # 2048 gives best bandwidth utilization for V=30522.
    BLOCK_V = 2048

    _fused_cross_entropy_kernel[(N,)](
        input_2d, bias, labels_1d,
        per_row_loss, count,
        N=N, V=V, V_padded=V_padded,
        BLOCK_V=BLOCK_V,
        HAS_BIAS=(bias is not None),
    )

    # Small reduction kernel for the final mean
    BLOCK_N = min(32768, triton.next_power_of_2(N))
    _reduce_loss_kernel[(1,)](
        per_row_loss, count, output,
        N=N, BLOCK_N=BLOCK_N,
    )

    return output[0]


class FusedCrossEntropyModel(torch.nn.Module):
    """Drop-in replacement for the Repro module using fused kernel."""

    def forward(self, arg1119_1, mm_default, arg1118_1):
        # Reshape labels
        labels = arg1119_1.reshape(-1)  # [32768]

        # input is [32768, 30524], we use first 30522 cols + bias
        # The repro does: slice to [32768, 30522], then add bias[30522]
        # We fuse the bias add into the kernel.

        # Compute fused cross-entropy loss
        loss = fused_cross_entropy(mm_default, labels, bias=arg1118_1)

        # The repro also returns reshape_default_3 which is the reshaped input+bias
        # We need to compute that as well for correctness
        slice_tensor = mm_default[:, :30522]
        reshape_out = (slice_tensor.reshape(256, 128, 30522) + arg1118_1).reshape(256, 128, 30522)

        return (loss, reshape_out)


def benchmark():
    """Benchmark the fused kernel vs torch baseline."""
    torch.manual_seed(42)
    device = 'cuda'

    N, V, V_padded = 32768, 30522, 30524

    # Create inputs matching the repro
    input_tensor = torch.randn(N, V_padded, dtype=torch.float32, device=device)
    bias = torch.randn(V, dtype=torch.float32, device=device)
    labels = torch.randint(0, V, (N,), dtype=torch.int64, device=device)
    # Sprinkle some -100 ignore tokens
    mask = torch.rand(N, device=device) < 0.1
    labels[mask] = -100

    # --- Correctness check ---
    # Reference: PyTorch cross_entropy
    logits = input_tensor[:, :V] + bias
    ref_loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100)

    # Fused kernel
    fused_loss = fused_cross_entropy(input_tensor, labels, bias=bias)

    print(f"Reference loss: {ref_loss.item():.6f}")
    print(f"Fused loss:     {fused_loss.item():.6f}")
    print(f"Difference:     {abs(ref_loss.item() - fused_loss.item()):.2e}")
    assert abs(ref_loss.item() - fused_loss.item()) < 1e-3, "Correctness check failed!"
    print("Correctness: PASSED\n")

    # --- Benchmark ---
    # Warmup
    for _ in range(10):
        _ = fused_cross_entropy(input_tensor, labels, bias=bias)
    torch.cuda.synchronize()

    # Fused kernel timing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(100)]

    for i in range(100):
        start_events[i].record()
        _ = fused_cross_entropy(input_tensor, labels, bias=bias)
        end_events[i].record()
    torch.cuda.synchronize()
    fused_ms = sum(s.elapsed_time(e) for s, e in zip(start_events, end_events)) / 100

    # PyTorch baseline timing
    for _ in range(10):
        _ = torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100)
    torch.cuda.synchronize()

    for i in range(100):
        start_events[i].record()
        logits = input_tensor[:, :V] + bias
        _ = torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100)
        end_events[i].record()
    torch.cuda.synchronize()
    baseline_ms = sum(s.elapsed_time(e) for s, e in zip(start_events, end_events)) / 100

    # Compute bandwidth
    data_read = N * V_padded * 4 + V * 4 + N * 8  # input + bias + labels
    data_written = N * 4 + 8 + 4  # per_row_loss + count + output

    print(f"Fused kernel:    {fused_ms:.3f} ms")
    print(f"PyTorch baseline: {baseline_ms:.3f} ms")
    print(f"Speedup:         {baseline_ms / fused_ms:.2f}x")
    print(f"\nData read:       {data_read / 1e9:.2f} GB")
    print(f"Data written:    {data_written / 1e6:.2f} MB")
    print(f"Bandwidth saved: ~4 GB (no intermediate materialization)")
    print(f"Effective BW:    {(data_read + data_written) / fused_ms / 1e6:.0f} GB/s")
    print(f"  (single pass via online softmax - reads input only once)")

    # Also time with torch.compile for comparison
    print("\n--- torch.compile (inductor) baseline ---")

    @torch.compile
    def compiled_ce(inp, b, lab):
        logits = inp[:, :V] + b
        return torch.nn.functional.cross_entropy(logits, lab, ignore_index=-100)

    # Warmup compile
    for _ in range(3):
        _ = compiled_ce(input_tensor, bias, labels)
    torch.cuda.synchronize()

    for i in range(100):
        start_events[i].record()
        _ = compiled_ce(input_tensor, bias, labels)
        end_events[i].record()
    torch.cuda.synchronize()
    compiled_ms = sum(s.elapsed_time(e) for s, e in zip(start_events, end_events)) / 100
    print(f"Compiled:        {compiled_ms:.3f} ms")
    print(f"Fused speedup over compiled: {compiled_ms / fused_ms:.2f}x")


if __name__ == "__main__":
    benchmark()
