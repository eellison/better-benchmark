"""cuTile port of pointwise_452ad66ee287: MobileBERT bf16 affine (scale + bias)."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _affine_bf16_kernel(
    input_ptr,
    scale_ptr,
    bias_ptr,
    output_ptr,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(input_ptr, index=(pid,), shape=(BLOCK,))
    # Row of length 128; broadcast scale/bias per row segment.
    # Load full scale/bias vectors of length 128 and tile via reshape/broadcast.
    # For BLOCK=2048 and inner dim 128, tile covers 16 rows.
    scale = ct.load(scale_ptr, index=(0,), shape=(128,))
    bias = ct.load(bias_ptr, index=(0,), shape=(128,))
    # Reshape input to (BLOCK//128, 128) then multiply by scale + bias.
    x_2d = ct.reshape(x, (BLOCK // 128, 128))
    x_f = ct.astype(x_2d, ct.float32)
    scale_2d = ct.reshape(scale, (1, 128))
    bias_2d = ct.reshape(bias, (1, 128))
    y = x_f * scale_2d + bias_2d
    y_bf = ct.astype(y, ct.bfloat16)
    y_flat = ct.reshape(y_bf, (BLOCK,))
    ct.store(output_ptr, index=(pid,), tile=y_flat)


@oracle_impl(hardware="B200", point="1398f333", BLOCK=2048)
def oracle_forward(inputs, *, BLOCK):
    arg0_1, arg1_1, arg2_1, _shape_param_0, _shape_param_1 = inputs
    final_shape = tuple(int(dim) for dim in _shape_param_1)
    output = torch.empty_strided(
        final_shape,
        (final_shape[1], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    # Flat views for the kernel
    numel = arg0_1.numel()
    x_flat = arg0_1.view(numel)
    out_flat = output.view(numel)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(numel, BLOCK), 1, 1),
        _affine_bf16_kernel,
        (x_flat, arg1_1, arg2_1, out_flat, BLOCK),
    )
    return output
