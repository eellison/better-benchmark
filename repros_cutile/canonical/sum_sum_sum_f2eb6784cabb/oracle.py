"""cuTile port of sum_sum_sum_f2eb6784cabb: VisFormer double-norm backward.

Complex multi-phase BN-backward-then-BN-backward-again pipeline. Torch handles
the numerous reductions and broadcast arithmetic; cuTile performs the two
elementwise epilogues that produce the final bf16 gradients.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BLOCK = 1024


def _make_epilogue_kernel():
    @ct.kernel
    def _epilogue(
        conv_ptr,        # f32 [NUMEL]
        sub_ptr,         # f32 [NUMEL]
        mean_term_ptr,   # f32 [C]
        dot_coeff_ptr,   # f32 [C]
        scale_ptr,       # f32 [C]
        channel_of_ptr,  # i32 [NUMEL]
        out_ptr,         # bf16 [NUMEL]
        BLOCK_: ct.Constant[int],
    ):
        pid = ct.bid(0)
        conv = ct.load(conv_ptr, index=(pid,), shape=(BLOCK_,))
        sub = ct.load(sub_ptr, index=(pid,), shape=(BLOCK_,))
        ch = ct.load(channel_of_ptr, index=(pid,), shape=(BLOCK_,))
        mt = ct.gather(mean_term_ptr, ch)
        dc = ct.gather(dot_coeff_ptr, ch)
        sc = ct.gather(scale_ptr, ch)
        adjusted = conv - sub * dc - mt
        out = adjusted * sc
        ct.store(out_ptr, index=(pid,), tile=ct.astype(out, ct.bfloat16))
    return _epilogue


_EPILOGUE_KERNEL = _make_epilogue_kernel()


def _bn_epilogue(conv_val, sub_val, mean_term, dot_coeff, scale, C, HW, NUMEL, device):
    n_range = torch.arange(NUMEL, device=device, dtype=torch.int32)
    channel_of = (n_range // HW) % C
    out_flat = torch.empty((NUMEL,), device=device, dtype=torch.bfloat16)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, ((NUMEL + BLOCK - 1) // BLOCK, 1, 1), _EPILOGUE_KERNEL,
        (conv_val.contiguous().view(-1),
         sub_val.contiguous().view(-1),
         mean_term, dot_coeff, scale, channel_of,
         out_flat, BLOCK),
    )
    return out_flat


def _run(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1 = inputs
    device = arg0_1.device
    N, C, H, W = arg0_1.shape
    HW = H * W
    NUMEL = N * C * HW
    # Repro hardcodes this constant across ALL shape points — it's the
    # reciprocal of (128 * 28 * 28) = 100352 baked into the graph. Later shape
    # points inherit the same constant even though the true HW differs.
    INV_R = 9.964923469387754e-06

    conv0 = arg0_1.to(torch.float32)
    sum_1 = conv0.sum(dim=[0, 2, 3])

    sub = arg1_1.to(torch.float32) - arg2_1
    mul = sub * arg3_1
    mul_1 = mul * arg4_1.view(1, C, 1, 1)
    add_val = mul_1 + arg5_1.view(1, C, 1, 1)
    conv_bf16 = add_val.to(torch.bfloat16)
    add_1 = conv_bf16.to(torch.float32) + arg6_1
    sub_1 = add_1 - arg7_1

    mul_2 = conv0 * sub_1
    sum_2 = mul_2.sum(dim=[0, 2, 3])

    mean_term_1 = sum_1 * INV_R
    dot_coeff_1 = (sum_2 * INV_R) * (arg8_1 * arg8_1)
    scale_1 = arg8_1 * arg9_1

    NUMEL_flat = N * C * HW
    dense1_flat = _bn_epilogue(
        conv0, sub_1, mean_term_1, dot_coeff_1, scale_1, C, HW, NUMEL_flat, device,
    )
    dense1 = dense1_flat.view(N, C, H, W)
    mul_10 = sum_2 * arg8_1

    # mul_9 in original is the f32 pre-epilogue-cast; the returned tensor is
    # convert_element_type_2. We reconstruct add_2 = arg10_1 + mul_9 in f32,
    # then downcast for the second-stage inputs. Rather than recomputing mul_9
    # in f32 here, use the epilogue's f32 view (before bf16 cast).
    mul_9_f32 = (conv0 - sub_1 * dot_coeff_1.view(1, C, 1, 1) - mean_term_1.view(1, C, 1, 1)) * scale_1.view(1, C, 1, 1)
    add_2 = arg10_1 + mul_9_f32
    conv_2_bf = add_2.to(torch.bfloat16)
    sum_3 = add_2.sum(dim=[0], keepdim=True)  # f32[1, 192, 28, 28]

    conv_2 = conv_2_bf.to(torch.float32)
    squeeze_a2 = arg2_1.view(C)
    sub_4 = arg1_1.to(torch.float32) - squeeze_a2.view(1, C, 1, 1)
    sum_4 = conv_2.sum(dim=[0, 2, 3])
    mul_11 = conv_2 * sub_4
    sum_5 = mul_11.sum(dim=[0, 2, 3])
    mean_term_2 = sum_4 * INV_R
    squeeze_a3 = arg3_1.view(C)
    dot_coeff_2 = (sum_5 * INV_R) * (squeeze_a3 * squeeze_a3)
    scale_2 = squeeze_a3 * arg4_1

    dense2_flat = _bn_epilogue(
        conv_2, sub_4, mean_term_2, dot_coeff_2, scale_2, C, HW, NUMEL_flat, device,
    )
    dense2 = dense2_flat.view(N, C, H, W)
    mul_19 = sum_5 * squeeze_a3

    sum_6 = dense2.sum(dim=[0, 2, 3]).to(torch.float32)

    return sum_1, mul_10, sum_3, sum_4, mul_19, dense2, sum_6


@oracle_impl(hardware="B200", point="ff30dd34")
@oracle_impl(hardware="B200", point="7a6295cd")
@oracle_impl(hardware="B200", point="fd9590cc")
def oracle_forward(inputs):
    return _run(inputs)
