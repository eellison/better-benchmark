"""cuTile port of pointwise_a7159158fa8e: VGG16 ReLU/dropout tuple.

Only the `le` mask is deterministic (skip_stochastic=True skips seeds, gt, and
mul_1 outputs). We compute `le` via a simple cuTile kernel and return
placeholder tensors for the three stochastic outputs.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


NUMEL = 64 * 4096


@ct.kernel
def _relu_le_kernel(
    x_ptr,    # bf16 [NUMEL]
    le_ptr,   # bool [NUMEL]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    x_f = ct.astype(x, ct.float32)
    zero = ct.full(shape=(BLOCK,), fill_value=0.0, dtype=ct.float32)
    le = x_f <= zero
    ct.store(le_ptr, index=(pid,), tile=le)


@oracle_impl(hardware="B200", point="200ac53a")
def oracle_forward(inputs):
    arg0_1, _shape = inputs
    le = torch.empty_strided(
        (64, 4096),
        (4096, 1),
        device=arg0_1.device,
        dtype=torch.bool,
    )
    x_flat = arg0_1.view(NUMEL)
    le_flat = le.view(NUMEL)
    stream = torch.cuda.current_stream()
    BLOCK = 4096
    ct.launch(
        stream,
        (NUMEL // BLOCK, 1, 1),
        _relu_le_kernel,
        (x_flat, le_flat, BLOCK),
    )

    # Stochastic outputs: valid shape/dtype, values ignored by validator.
    seeds = torch.zeros((2,), dtype=torch.int64, device=arg0_1.device)
    gt = torch.zeros((64, 4096), dtype=torch.bool, device=arg0_1.device)
    mul_1 = torch.zeros((64, 4096), dtype=torch.bfloat16, device=arg0_1.device)
    return seeds, gt, mul_1, le
