"""cuTile port of pointwise_2b567494bf14 (SCHEDULER_FUSION): Longformer bias-add,
divide-by-8, and 3-window overlap layout materialization for [288,512,64].
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 8
HEAD = 12
FEATURE_PER_HEAD = 64
HIDDEN = HEAD * FEATURE_PER_HEAD  # 768
POS_PER_WINDOW = 512
WINDOW_STEP = 256  # stride between window starts
WINDOWS = 3
CHUNKS = BATCH * HEAD * WINDOWS  # 288


@ct.kernel
def _longformer_stencil_kernel(
    x_ptr,     # bf16 [1024, BATCH, HIDDEN]
    bias_ptr,  # bf16 [HIDDEN]
    out_ptr,   # bf16 [CHUNKS, POS_PER_WINDOW, FEATURE_PER_HEAD]
    POS_PER_WINDOW_C: ct.Constant[int],
    FEATURE_PER_HEAD_C: ct.Constant[int],
    HEAD_C: ct.Constant[int],
    WINDOW_STEP_C: ct.Constant[int],
    WINDOWS_C: ct.Constant[int],
):
    chunk = ct.bid(0)
    # decompose chunk into (batch, head, window)
    head_batch = chunk // WINDOWS_C
    window = chunk - head_batch * WINDOWS_C
    batch = head_batch // HEAD_C
    head = head_batch - batch * HEAD_C

    # load bias for this head [FEATURE_PER_HEAD]
    bias = ct.load(bias_ptr, index=(head,), shape=(FEATURE_PER_HEAD_C,))
    bias_f = ct.astype(bias, ct.float32)

    # load a (POS_PER_WINDOW, FEATURE_PER_HEAD) tile from x
    # x is (1024, BATCH, HIDDEN), we need x[window*WINDOW_STEP:window*WINDOW_STEP+POS_PER_WINDOW, batch, head*FEATURE_PER_HEAD:(head+1)*FEATURE_PER_HEAD]
    # This is a strided view. cuTile doesn't do arbitrary gathers cheaply.
    # We can iterate over rows via nested tiles: for each of POS_PER_WINDOW rows,
    # load the FEATURE_PER_HEAD slice. But that's a loop.
    # Simpler: launch grid also over positions, so each program covers one row.
    # But this kernel doesn't - it's a single-program approach.
    # Given cuTile limits, we'll launch a 2D grid: (chunk, pos)
    pass


# Since the stride pattern is awkward for cuTile, use a simpler per-row kernel:
@ct.kernel
def _longformer_row_kernel(
    x_ptr,     # bf16 [1024, BATCH, HIDDEN]
    bias_ptr,  # bf16 [HIDDEN]
    out_ptr,   # bf16 [CHUNKS, POS_PER_WINDOW, FEATURE_PER_HEAD]
    POS_PER_WINDOW_C: ct.Constant[int],
    FEATURE_PER_HEAD_C: ct.Constant[int],
    HEAD_C: ct.Constant[int],
    WINDOW_STEP_C: ct.Constant[int],
    WINDOWS_C: ct.Constant[int],
):
    chunk = ct.bid(0)
    pos = ct.bid(1)

    head_batch = chunk // WINDOWS_C
    window = chunk - head_batch * WINDOWS_C
    batch = head_batch // HEAD_C
    head = head_batch - batch * HEAD_C

    src_row = window * WINDOW_STEP_C + pos
    # Load x[src_row, batch, head*FEATURE_PER_HEAD:(head+1)*FEATURE_PER_HEAD]
    # x is 3D [1024, BATCH, HIDDEN]. Load tile shape (1, 1, FEATURE_PER_HEAD)
    # at index (src_row, batch, head)
    x = ct.load(x_ptr, index=(src_row, batch, head), shape=(1, 1, FEATURE_PER_HEAD_C))
    x = ct.reshape(x, (FEATURE_PER_HEAD_C,))
    # Load bias for this head — bias is 2D [HEAD, FEATURE_PER_HEAD]
    bias = ct.load(bias_ptr, index=(head, 0), shape=(1, FEATURE_PER_HEAD_C))
    bias = ct.reshape(bias, (FEATURE_PER_HEAD_C,))
    x_f = ct.astype(x, ct.float32)
    bias_f = ct.astype(bias, ct.float32)
    added = ct.astype(ct.astype(x_f + bias_f, ct.bfloat16), ct.float32)
    scaled = ct.astype(added * 0.125, ct.bfloat16)
    scaled_3d = ct.reshape(scaled, (1, 1, FEATURE_PER_HEAD_C))
    ct.store(out_ptr, index=(chunk, pos, 0), tile=scaled_3d)


@oracle_impl(hardware="B200", point="5fa3702b", BLOCK_P=32, BLOCK_D=64)
def oracle_forward(inputs, *, BLOCK_P: int, BLOCK_D: int):
    arg0, arg1, _shape0, _shape1, _shape2, _shape3, _shape4, _stride4, shape5 = inputs
    out = torch.empty_strided(
        tuple(int(dim) for dim in shape5),
        (POS_PER_WINDOW * FEATURE_PER_HEAD, FEATURE_PER_HEAD, 1),
        device=arg0.device,
        dtype=arg0.dtype,
    )
    # Reshape input from [8192, 768] to [1024, BATCH, HIDDEN]
    x_3d = arg0.view(1024, BATCH, HIDDEN)
    # Reshape bias to allow per-head indexing: bias is [HIDDEN=768], view as [HEAD, FEATURE_PER_HEAD]
    bias_2d = arg1.view(HEAD, FEATURE_PER_HEAD)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (CHUNKS, POS_PER_WINDOW, 1),
        _longformer_row_kernel,
        (x_3d, bias_2d, out, POS_PER_WINDOW, FEATURE_PER_HEAD, HEAD, WINDOW_STEP, WINDOWS),
    )
    return out
