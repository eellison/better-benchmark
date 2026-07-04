"""cuTile port of pointwise_8bb40810003c: f32->bf16 cast with permute view alias.

ALGEBRAIC_ELIMINATION: cast the fp32 input to bf16 into one contiguous
[rows, cols] buffer and return two views — the permuted transpose and the
original layout — as aliases of that buffer. cuTile stores whole tiles with
no mask, so we pick the largest power-of-2 BLOCK (up to 1024) that divides
the total element count for each shape point.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _f32_to_bf16_kernel(
    x_ptr,     # f32 [N]
    out_ptr,   # bf16 [N]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    ct.store(out_ptr, index=(pid,), tile=ct.astype(x, ct.bfloat16))


_POINTS = [
    "0cf39344", "ce1e0a23", "ae9f1068", "44a2434c", "f5f85987", "e24f40e4",
    "9e822d8f", "2e98be1c", "5cffd879", "e1aec24f", "aafa641b", "2613646b",
    "d29f2cfb", "5617a165", "ac612a70", "f0eaed89", "0a486f34", "ebcabf6d",
    "debf1fc5", "b30c1d04", "6c896102", "18173249", "62980a68", "af8b80d5",
    "9f2d4100", "1b40fbb6", "557311da", "5c0408c4", "b1ca53d6", "a1cb3860",
    "52d3c6a7", "01baa4eb", "43065864", "16e5c8c0", "fdcbaff1", "25dff7ef",
    "7cfcfe39", "19d8a989", "17dcd71b", "c4cfc269", "fb817d2a", "2d6d0114",
    "32e09daf", "e340bbdf", "1185642c", "d84d33a1", "af634b6e", "46c132a4",
    "8cf9b83f", "16d2cbc0", "62c6abfb", "595a24d8", "ec623101", "dc3e8b06",
    "040e49cd", "3798a6aa", "8aa63c17", "3851a1ae", "945f2c72", "6747b18a",
    "e692d575", "688f3d95", "559a6c52", "68efa070", "1c987e3e", "e583d54f",
    "2d7cdfa7", "e56e2d8f", "b016ff56", "2bb61127", "7a086584", "fdf8e815",
    "76ad9430", "b3e78a1b", "4f207a72", "58e39a16", "f4e93827", "60ceae96",
    "8d573313", "aa459734", "a7aca999", "a876c3bb", "afe477f0", "7b445016",
]


def _apply_decorators(fn):
    for p in _POINTS:
        fn = oracle_impl(hardware="B200", point=p)(fn)
    return fn


def _largest_pow2_divisor(n, cap=1024):
    """Largest power of 2 <= cap that divides n."""
    b = cap
    while b > 1 and (n % b) != 0:
        b //= 2
    return b


@_apply_decorators
def oracle_forward(inputs):
    arg0_1 = inputs[0]
    rows, cols = arg0_1.shape
    base = torch.empty_strided(
        (int(rows), int(cols)),
        (int(cols), 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    n_elements = int(arg0_1.numel())
    x_flat = arg0_1.reshape(n_elements)
    base_flat = base.view(n_elements)
    BLOCK = _largest_pow2_divisor(n_elements, cap=1024)
    grid = (n_elements // BLOCK, 1, 1)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        grid,
        _f32_to_bf16_kernel,
        (x_flat, base_flat, BLOCK),
    )
    return base.permute(1, 0), base
