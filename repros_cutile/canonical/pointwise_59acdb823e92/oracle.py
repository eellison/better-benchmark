"""cuTile port of pointwise_59acdb823e92: MoE routing postprocess.

The Triton oracle used custom kernels for the metadata (invalid mask + int32
cast) and the sorted-row gather. cuTile's `load` needs compile-time index
coordinates, so the runtime-indexed gather is delegated to torch (matching the
reference oracle's use of `torch.topk` and `torch.sort` for the ordering
boundary anyway). The remaining pointwise work (invalid mask, int32 cast) is
straightforward.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 1000
EXPERTS = 32
TOPK = 4
TOTAL_TOPK = ROWS * TOPK
HIDDEN = 2880


@ct.kernel
def _routing_metadata_kernel(
    sorted_ids_ptr,  # (TOTAL_TOPK,) i64
    mask_ptr,        # (TOTAL_TOPK,) bool
    cast_ptr,        # (TOTAL_TOPK,) i32
    BLOCK: ct.Constant[int],
    VALID_EXPERTS: ct.Constant[int],
):
    pid = ct.bid(0)
    ids = ct.load(sorted_ids_ptr, index=(pid,), shape=(BLOCK,))
    threshold = ct.full(shape=(BLOCK,), fill_value=VALID_EXPERTS, dtype=ct.int64)
    invalid = ids >= threshold
    ct.store(mask_ptr, index=(pid,), tile=invalid)
    ct.store(cast_ptr, index=(pid,), tile=ct.astype(ids, ct.int32))


@oracle_impl(
    hardware="B200",
    point="a9d89d2e",
    BLOCK_META=1024,
)
def oracle_forward(inputs, *, BLOCK_META):
    arg0_1, arg1_1 = inputs

    topk_values, topk_indices = torch.topk(arg0_1, TOPK)
    flat_indices = topk_indices.view(-1)
    sorted_ids, sort_positions = torch.sort(flat_indices)

    # Host-side gather and mask.
    div_index = torch.div(sort_positions, TOPK, rounding_mode="floor")
    gathered = arg1_1[div_index]
    unsqueeze = torch.empty((TOTAL_TOPK, 1), device=arg0_1.device, dtype=torch.bool)
    cast = torch.empty((TOTAL_TOPK,), device=arg0_1.device, dtype=torch.int32)

    # BLOCK_META must divide TOTAL_TOPK (4000)? 1024 doesn't divide 4000.
    # cuTile doesn't support masked stores, so use a divisor: TOTAL_TOPK=4000
    # = 2^5 * 5^3. Use BLOCK=8 (largest power of 2 dividing 4000: 2^5 = 32).
    # Actually 4000/32 = 125, so BLOCK=32 works. But cuTile requires tile
    # shapes to be a power of 2 — 32 is fine.
    block = 1
    while block * 2 <= BLOCK_META and TOTAL_TOPK % (block * 2) == 0:
        block *= 2

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (TOTAL_TOPK // block, 1, 1),
        _routing_metadata_kernel,
        (sorted_ids, unsqueeze.view(-1), cast, block, EXPERTS),
    )

    # Where the mask is set, zero out the corresponding row. Since we're
    # doing this on host it's a simple where broadcast.
    zero = torch.zeros((), device=arg1_1.device, dtype=torch.bfloat16)
    where = torch.where(unsqueeze, zero, gathered)

    return topk_values, sorted_ids, sort_positions, unsqueeze, where, cast
