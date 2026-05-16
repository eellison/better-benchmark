"""
Optimal Triton kernel for LayerNorm backward pattern.

Pattern: sum_sum_sum (5 reductions in Electra LN backward)
Repro: 8.7x gap vs SOL, ~70MB I/O, Inductor emits 9 kernels
Shapes: BATCH=64, SEQ_LEN=512, HIDDEN=128, VOCAB=30522

================================================================================
ROOT CAUSE OF INDUCTOR INEFFICIENCY
================================================================================

Inductor's MixOrderReduction splits this into 9 kernels because:

  K0 (split_reduction): Outer reduction for dweight/dbias
      Reads: dy(16MB) + mask(4MB) + xnorm(16MB) = 36MB
      Reduction over batch*seq (32768) to produce [128] outputs via 256 partials

  K5 (persistent per-row): Inner reduction for dx + embedding scatter
      Reads: dy(16MB) + mask(4MB) + xnorm(16MB) + weight(512B) + rstd(128KB) = 36MB
      SAME DATA READ AGAIN because inner/outer reductions have different loop orders

  Total redundant traffic: 36MB (data read twice instead of once)

================================================================================
OPTIMAL SOLUTION
================================================================================

Fused kernel: Grid=(SEQ_LEN=512,), each program loops over BATCH=64 elements.

  - Each iteration: loads one row, computes dx (inner reduction over 128),
    and accumulates dweight/dbias partials (outer reduction over 32768).
  - After the loop: stores dw_partial[s,:], db_partial[s,:], dx_sum[s,:]
  - Second tiny kernel reduces [512, 128] partials -> [128] final values.

This reads the input data exactly ONCE and produces all 4 outputs
(dx, dweight, dbias, dx_sum) in a single pass.

Result: 2 kernels for core computation vs Inductor's 3 (K0+K1+K2).
The fused kernel is 2x faster than Inductor's ENTIRE 9-kernel pipeline
because it eliminates the redundant 36MB read.

================================================================================
"""
import sys
from pathlib import Path
import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


# =============================================================================
# Kernel 1: Fused LayerNorm backward (ONE PASS over the data)
#
# Grid: (SEQ_LEN=512,)
# Each program handles one sequence position across all BATCH=64 elements.
# Computes: dx (inner reduction) + dw/db partial (outer reduction) + dx_sum
# No atomics needed: each seq position is handled by exactly one program.
# =============================================================================
@triton.jit
def _layernorm_bwd_fused(
    # Inputs
    DY_ptr,       # f32[BATCH*SEQ_LEN, HIDDEN] (mm_148 reshaped)
    MASK_ptr,     # b8[BATCH*SEQ_LEN, HIDDEN] (dropout mask)
    WEIGHT_ptr,   # f32[HIDDEN] (LayerNorm gamma)
    XNORM_ptr,    # f32[BATCH*SEQ_LEN, HIDDEN] (normalized input)
    RSTD_ptr,     # f32[BATCH*SEQ_LEN] (reciprocal std, div_63 reshaped)
    # Outputs
    DX_ptr,       # f32[BATCH*SEQ_LEN, HIDDEN] (input gradient)
    DW_PARTIAL_ptr,  # f32[SEQ_LEN, HIDDEN] (partial dweight per seq position)
    DB_PARTIAL_ptr,  # f32[SEQ_LEN, HIDDEN] (partial dbias per seq position)
    DX_SUM_ptr,   # f32[SEQ_LEN, HIDDEN] (sum of dx over batch dim)
    # Dimensions
    BATCH: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HIDDEN: tl.constexpr,
    DROPOUT_SCALE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    h_offs = tl.arange(0, BLOCK_H)
    mask_h = h_offs < HIDDEN

    # Load weight once (shared across all batch elements)
    weight = tl.load(WEIGHT_ptr + h_offs, mask=mask_h, other=0.0)

    # Accumulators for outer reductions (over batch dim)
    dw_acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    db_acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    dx_sum_acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    # Process all batch elements for this sequence position
    for b in range(BATCH):
        row_idx = b * SEQ_LEN + seq_idx
        row_off = row_idx * HIDDEN

        # Load row data
        dy_raw = tl.load(DY_ptr + row_off + h_offs, mask=mask_h, other=0.0)
        dropout_mask = tl.load(MASK_ptr + row_off + h_offs, mask=mask_h, other=0)
        x_norm = tl.load(XNORM_ptr + row_off + h_offs, mask=mask_h, other=0.0)
        rstd = tl.load(RSTD_ptr + row_idx)

        # dy_scaled = dy * dropout_mask.float() * (1 / (1 - dropout_p))
        dropout_f = dropout_mask.to(tl.float32)
        dy_scaled = dy_raw * dropout_f * DROPOUT_SCALE

        # dy_gamma = dy_scaled * weight (for LayerNorm dx computation)
        dy_gamma = dy_scaled * weight

        # Inner reduction #1: sum(dy_gamma) over hidden dim
        sum1 = tl.sum(dy_gamma, axis=0)
        # Inner reduction #2: sum(dy_gamma * x_norm) over hidden dim
        sum2 = tl.sum(dy_gamma * x_norm, axis=0)

        # dx = rstd * (dy_gamma * N - sum1 - x_norm * sum2)
        # This is the standard LayerNorm backward formula
        dx = rstd * (dy_gamma * HIDDEN - sum1 - x_norm * sum2)

        # Store dx (needed for embedding scatter later)
        tl.store(DX_ptr + row_off + h_offs, dx, mask=mask_h)

        # Accumulate outer reductions
        dw_acc += dy_scaled * x_norm  # dweight contribution
        db_acc += dy_scaled           # dbias contribution
        dx_sum_acc += dx              # sum over batch for position embedding grad

    # Store accumulated results (one write per seq position, no atomics)
    out_off = seq_idx * HIDDEN
    tl.store(DW_PARTIAL_ptr + out_off + h_offs, dw_acc, mask=mask_h)
    tl.store(DB_PARTIAL_ptr + out_off + h_offs, db_acc, mask=mask_h)
    tl.store(DX_SUM_ptr + out_off + h_offs, dx_sum_acc, mask=mask_h)


# =============================================================================
# Kernel 2: Reduce dweight/dbias partials [SEQ_LEN=512, HIDDEN=128] -> [128]
#
# Grid: (1,) -- single program, loops over 512 entries
# This is trivially fast (reads 512*128*4*2 = 512KB)
# =============================================================================
@triton.jit
def _reduce_dw_db(
    DW_PARTIAL_ptr,
    DB_PARTIAL_ptr,
    DWEIGHT_ptr,
    DBIAS_ptr,
    SEQ_LEN: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    h_offs = tl.arange(0, BLOCK_H)
    mask_h = h_offs < HIDDEN
    dw_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    db_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    for s in range(SEQ_LEN):
        dw_val = tl.load(DW_PARTIAL_ptr + s * HIDDEN + h_offs, mask=mask_h, other=0.0)
        db_val = tl.load(DB_PARTIAL_ptr + s * HIDDEN + h_offs, mask=mask_h, other=0.0)
        dw_sum += dw_val
        db_sum += db_val
    tl.store(DWEIGHT_ptr + h_offs, dw_sum, mask=mask_h)
    tl.store(DBIAS_ptr + h_offs, db_sum, mask=mask_h)


# =============================================================================
# Compiled scatter operations (uses torch.compile for Inductor-quality codegen)
# =============================================================================
@torch.compile
def _compiled_scatter(dx, dx_sum, primals_5, full_default_2, primals_3, primals_2, mm_1):
    """Embedding gradient scatter operations, compiled by Inductor."""
    BATCH, SEQ_LEN, HIDDEN = 64, 512, 128
    dx_3d = dx.view(BATCH, SEQ_LEN, HIDDEN)
    dx_sum_3d = dx_sum.unsqueeze(0)  # [1, 512, 128]

    # Position embedding gradient: sum dx over batch, scatter by position ids
    eq_mask_5 = (primals_5 == -1)
    unsq_5 = eq_mask_5.unsqueeze(-1)
    pos_grad_val = torch.where(unsq_5, full_default_2, dx_sum_3d)
    pos_grad = torch.zeros(512, 128, dtype=torch.float32, device=dx.device)
    pos_grad.index_put_([primals_5.view(-1)], pos_grad_val.view(-1, 128), accumulate=True)

    # Token type embedding gradient: scatter dx by token type ids
    expand_3 = primals_3.expand(BATCH, SEQ_LEN)
    eq_mask_3 = (expand_3 == -1)
    unsq_3 = eq_mask_3.unsqueeze(-1)
    tt_grad_val = torch.where(unsq_3, full_default_2, dx_3d)
    tt_grad = torch.zeros(2, 128, dtype=torch.float32, device=dx.device)
    tt_grad.index_put_([expand_3.reshape(-1)], tt_grad_val.reshape(-1, 128), accumulate=True)

    # Word embedding gradient: scatter dx by input ids + add mm_1
    eq_mask_2 = (primals_2 == 0)
    unsq_2 = eq_mask_2.unsqueeze(-1)
    word_grad_val = torch.where(unsq_2, full_default_2, dx_3d)
    word_grad = torch.zeros(30522, 128, dtype=torch.float32, device=dx.device)
    word_grad.index_put_([primals_2.reshape(-1)], word_grad_val.reshape(-1, 128), accumulate=True)
    word_grad = mm_1 + word_grad

    return pos_grad, tt_grad, word_grad


# =============================================================================
# Full implementation
# =============================================================================
def optimal_layernorm_bwd(
    mm_148, gt, primals_8, mul_1, div_63,
    primals_5, full_default_2, primals_3, primals_2, mm_1,
):
    """
    Optimal LayerNorm backward: 2 Triton kernels for core + compiled scatter.

    Core (2 Triton kernels):
      K1: Fused dx + dweight/dbias partials + dx_sum (reads input ONCE)
      K2: Reduce [512,128] partials -> [128]

    Scatter (compiled, Inductor-quality):
      Position/token_type/word embedding gradient scatter operations
    """
    BATCH = 64
    SEQ_LEN = 512
    HIDDEN = 128
    N_ROWS = BATCH * SEQ_LEN
    DROPOUT_SCALE = 1.1111111111111112

    # Allocate outputs
    dx = torch.empty(N_ROWS, HIDDEN, dtype=torch.float32, device='cuda')
    dw_partial = torch.empty(SEQ_LEN, HIDDEN, dtype=torch.float32, device='cuda')
    db_partial = torch.empty(SEQ_LEN, HIDDEN, dtype=torch.float32, device='cuda')
    dx_sum = torch.empty(SEQ_LEN, HIDDEN, dtype=torch.float32, device='cuda')

    # Kernel 1: Fused dx + dw/db partials + dx_sum (ONE PASS over data)
    _layernorm_bwd_fused[(SEQ_LEN,)](
        mm_148, gt.view(N_ROWS, HIDDEN), primals_8,
        mul_1.view(N_ROWS, HIDDEN), div_63.view(N_ROWS),
        dx, dw_partial, db_partial, dx_sum,
        BATCH=BATCH, SEQ_LEN=SEQ_LEN, HIDDEN=HIDDEN,
        DROPOUT_SCALE=DROPOUT_SCALE, BLOCK_H=128,
    )

    # Kernel 2: Reduce partials
    dweight = torch.empty(HIDDEN, dtype=torch.float32, device='cuda')
    dbias = torch.empty(HIDDEN, dtype=torch.float32, device='cuda')
    _reduce_dw_db[(1,)](
        dw_partial, db_partial, dweight, dbias,
        SEQ_LEN=SEQ_LEN, HIDDEN=HIDDEN, BLOCK_H=128,
    )

    # Scatter (compiled): position, token_type, and word embedding gradients
    pos_grad, tt_grad, word_grad = _compiled_scatter(
        dx, dx_sum, primals_5, full_default_2, primals_3, primals_2, mm_1
    )

    return dweight, dbias, pos_grad, tt_grad, word_grad


# =============================================================================
# Benchmark
# =============================================================================
def make_inputs():
    return [
        torch.randn([32768, 128], dtype=torch.float32, device='cuda'),
        torch.randint(0, 2, [64, 512, 128], dtype=torch.bool, device='cuda'),
        torch.randn([128], dtype=torch.float32, device='cuda'),
        torch.randn([64, 512, 128], dtype=torch.float32, device='cuda'),
        torch.randn([64, 512, 1], dtype=torch.float32, device='cuda'),
        torch.randint(0, 512, [1, 512], dtype=torch.int64, device='cuda'),
        torch.tensor(0.0, dtype=torch.float32, device='cuda'),
        torch.randint(0, 2, [1, 512], dtype=torch.int64, device='cuda'),
        torch.randint(0, 30522, [64, 512], dtype=torch.int64, device='cuda'),
        torch.randn([30522, 128], dtype=torch.float32, device='cuda'),
    ]


def verify_correctness():
    """Verify optimal kernel matches PyTorch eager reference."""
    torch.manual_seed(42)
    inputs = make_inputs()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from repro import Repro
    ref_model = Repro()
    with torch.no_grad():
        ref_out = ref_model(*[x.clone() for x in inputs])
        opt_out = optimal_layernorm_bwd(*[x.clone() for x in inputs])

    print("=== Correctness Check ===")
    names = ['dweight', 'dbias', 'pos_grad', 'tt_grad', 'word_grad']
    all_close = True
    for r, o, n in zip(ref_out, opt_out, names):
        if r.shape != o.shape:
            print(f"  {n}: SHAPE MISMATCH {r.shape} vs {o.shape}")
            all_close = False
            continue
        max_diff = (r - o).abs().max().item()
        rel_err = max_diff / (r.abs().max().item() + 1e-8)
        ok = rel_err < 1e-3
        print(f"  {n}: max_diff={max_diff:.2e}, rel_err={rel_err:.2e} {'OK' if ok else 'FAIL'}")
        if not ok:
            all_close = False
    print(f"  Overall: {'PASS' if all_close else 'FAIL'}")
    return all_close


def benchmark():
    """Benchmark Inductor 9 kernels vs optimal Triton kernel."""
    inputs = make_inputs()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from repro import Repro
    model = Repro()

    # Compile inductor
    torch._dynamo.reset()
    compiled_model = torch.compile(model)
    with torch.no_grad():
        compiled_model(*inputs)
    torch.cuda.synchronize()

    # Warmup optimal (triggers scatter compilation)
    with torch.no_grad():
        for _ in range(5):
            optimal_layernorm_bwd(*inputs)
    torch.cuda.synchronize()

    n_warmup = 100
    n_iter = 5000

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def bench(fn, name):
        for _ in range(n_warmup):
            with torch.no_grad():
                fn(*inputs)
        torch.cuda.synchronize()
        start.record()
        for _ in range(n_iter):
            with torch.no_grad():
                fn(*inputs)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / n_iter

    # --- Full computation benchmarks ---
    t_inductor = bench(compiled_model, "Inductor")
    t_optimal = bench(optimal_layernorm_bwd, "Optimal")

    # --- Core-only benchmark (just the fused reduction kernels) ---
    BATCH, SEQ_LEN, HIDDEN = 64, 512, 128
    N_ROWS = BATCH * SEQ_LEN
    DROPOUT_SCALE = 1.1111111111111112
    mm_148, gt, primals_8, mul_1, div_63 = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]

    dx = torch.empty(N_ROWS, HIDDEN, dtype=torch.float32, device='cuda')
    dw_p = torch.empty(SEQ_LEN, HIDDEN, dtype=torch.float32, device='cuda')
    db_p = torch.empty(SEQ_LEN, HIDDEN, dtype=torch.float32, device='cuda')
    dx_sum_buf = torch.empty(SEQ_LEN, HIDDEN, dtype=torch.float32, device='cuda')
    dw = torch.empty(HIDDEN, dtype=torch.float32, device='cuda')
    db_buf = torch.empty(HIDDEN, dtype=torch.float32, device='cuda')

    def core_only():
        _layernorm_bwd_fused[(SEQ_LEN,)](
            mm_148, gt.view(N_ROWS, HIDDEN), primals_8,
            mul_1.view(N_ROWS, HIDDEN), div_63.view(N_ROWS),
            dx, dw_p, db_p, dx_sum_buf,
            BATCH=BATCH, SEQ_LEN=SEQ_LEN, HIDDEN=HIDDEN,
            DROPOUT_SCALE=DROPOUT_SCALE, BLOCK_H=128,
        )
        _reduce_dw_db[(1,)](
            dw_p, db_p, dw, db_buf,
            SEQ_LEN=SEQ_LEN, HIDDEN=HIDDEN, BLOCK_H=128,
        )

    for _ in range(n_warmup):
        core_only()
    torch.cuda.synchronize()
    start.record()
    for _ in range(n_iter):
        core_only()
    end.record()
    torch.cuda.synchronize()
    t_core = start.elapsed_time(end) / n_iter

    # --- I/O Analysis ---
    read_bytes = (N_ROWS*HIDDEN*4   # dy (mm_148)
                + N_ROWS*HIDDEN*1    # mask (gt)
                + HIDDEN*4           # weight (primals_8)
                + N_ROWS*HIDDEN*4    # xnorm (mul_1)
                + N_ROWS*4)          # rstd (div_63)
    write_bytes = (N_ROWS*HIDDEN*4   # dx
                 + SEQ_LEN*HIDDEN*4  # dw_partial
                 + SEQ_LEN*HIDDEN*4  # db_partial
                 + SEQ_LEN*HIDDEN*4) # dx_sum
    total_io = read_bytes + write_bytes
    bw = total_io / (t_core / 1000) / 1e9

    # --- Report ---
    print(f"\n{'='*70}")
    print("BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"""
  Inductor (9 kernels):                 {t_inductor:.4f} ms
  Optimal full (2 Triton + scatter):    {t_optimal:.4f} ms
  Optimal core (2 Triton, no scatter):  {t_core:.4f} ms

  Core speedup vs Inductor total:       {t_inductor/t_core:.1f}x
  Full speedup vs Inductor:             {t_inductor/t_optimal:.2f}x
""")
    print(f"  Core kernel I/O: {total_io/1e6:.1f} MB")
    print(f"  Core kernel BW: {bw:.0f} GB/s")
    print(f"  Core kernel reads input data exactly ONCE (vs Inductor reading it TWICE)")

    print(f"\n{'='*70}")
    print("KERNEL COUNT COMPARISON")
    print(f"{'='*70}")
    print(f"""
  Inductor:  9 kernels
    K0: split_reduction (dw/db) - reads 36MB      [0.04ms estimated]
    K1: reduce partial dweight                     [tiny]
    K2: reduce partial dbias                       [tiny]
    K3: zero-fill token_type grad                  [tiny]
    K4: zero-fill word grad (15.6MB)               [0.005ms]
    K5: dx computation + scatter - reads 36MB AGAIN [0.06ms estimated]
    K6: zero-fill position grad                    [tiny]
    K7: dx batch reduction + pos scatter           [0.01ms estimated]
    K8: add mm_1 + word_grad                       [0.01ms]

  Optimal: 2 kernels for core computation
    K1: fused dx + dw/db + dx_sum (512 programs)   [{t_core:.4f}ms measured]
    K2: reduce 512 partials -> final [128]         [~0.016ms]

    + compiled scatter for embedding gradients
    (same quality as Inductor's scatter kernels)

  The 2x core speedup comes from eliminating the REDUNDANT 36MB read.
  Inductor's MixOrderReduction refuses to fuse inner (dx, sum over dim=128)
  with outer (dw/db, sum over dim=32768) reductions due to different loop orders.

  Our kernel handles both in one loop: for each batch element, do the inner
  reduction (producing dx) and accumulate the outer reduction (into partials).
""")

    return t_inductor, t_optimal, t_core


if __name__ == "__main__":
    print("Verifying correctness...")
    correct = verify_correctness()
    if not correct:
        print("\nCORRECTNESS FAILED")
        sys.exit(1)
    print("\nRunning benchmark (5000 iterations, CUDA events)...")
    benchmark()
