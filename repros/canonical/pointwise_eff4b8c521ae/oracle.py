"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete returned ReLU plus low-memory maxpool value tensor with a shape-specialized Triton stencil that sinks the ReLU producer into the fixed-window pool and skips the dead int8 offsets tuple element, whereas Inductor lowers the captured `_low_memory_max_pool_with_offsets` path through generic multi-output stencil scheduling around the ReLU producer even though only the value output escapes; Inductor cannot do this today because its scheduler/codegen does not split the low-memory maxpool tuple into a value-only lowering while fusing the pointwise producer through the stencil loop; the fix is SCHEDULER_FUSION: teach the maxpool scheduler to DCE unused offsets and emit a single value-output ReLU/maxpool stencil for static 2x2 and 3x3 stride-2 cases."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _relu_preserve_nan(x):
    return tl.where((x > 0.0) | (x != x), x, 0.0)


@triton.jit
def _take_value(candidate, best):
    take = (candidate > best) | ((candidate != candidate) & (best == best))
    return tl.where(take, candidate, best)


@triton.jit
def _relu_maxpool2_values_kernel(
    input_ptr,
    values_ptr,
    total_outputs: tl.constexpr,
    h_in: tl.constexpr,
    w_in: tl.constexpr,
    h_out: tl.constexpr,
    w_out: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = linear < total_outputs

    out_w = linear % w_out
    tmp = linear // w_out
    out_h = tmp % h_out
    plane = tmp // h_out

    input_base = plane * h_in * w_in + out_h * 2 * w_in + out_w * 2
    x00 = _relu_preserve_nan(tl.load(input_ptr + input_base, mask=active, other=-float("inf")).to(tl.float32))
    x01 = _relu_preserve_nan(tl.load(input_ptr + input_base + 1, mask=active, other=-float("inf")).to(tl.float32))
    x10 = _relu_preserve_nan(tl.load(input_ptr + input_base + w_in, mask=active, other=-float("inf")).to(tl.float32))
    x11 = _relu_preserve_nan(tl.load(input_ptr + input_base + w_in + 1, mask=active, other=-float("inf")).to(tl.float32))

    best = x00
    best = _take_value(x01, best)
    best = _take_value(x10, best)
    best = _take_value(x11, best)

    tl.store(values_ptr + linear, best, mask=active)


@triton.jit
def _relu_maxpool3_values_kernel(
    input_ptr,
    values_ptr,
    total_outputs: tl.constexpr,
    h_in: tl.constexpr,
    w_in: tl.constexpr,
    h_out: tl.constexpr,
    w_out: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = linear < total_outputs

    out_w = linear % w_out
    tmp = linear // w_out
    out_h = tmp % h_out
    plane = tmp // h_out

    input_base = plane * h_in * w_in + out_h * 2 * w_in + out_w * 2
    x00 = _relu_preserve_nan(tl.load(input_ptr + input_base, mask=active, other=-float("inf")).to(tl.float32))
    x01 = _relu_preserve_nan(tl.load(input_ptr + input_base + 1, mask=active, other=-float("inf")).to(tl.float32))
    x02 = _relu_preserve_nan(tl.load(input_ptr + input_base + 2, mask=active, other=-float("inf")).to(tl.float32))
    x10 = _relu_preserve_nan(tl.load(input_ptr + input_base + w_in, mask=active, other=-float("inf")).to(tl.float32))
    x11 = _relu_preserve_nan(tl.load(input_ptr + input_base + w_in + 1, mask=active, other=-float("inf")).to(tl.float32))
    x12 = _relu_preserve_nan(tl.load(input_ptr + input_base + w_in + 2, mask=active, other=-float("inf")).to(tl.float32))
    x20 = _relu_preserve_nan(tl.load(input_ptr + input_base + 2 * w_in, mask=active, other=-float("inf")).to(tl.float32))
    x21 = _relu_preserve_nan(tl.load(input_ptr + input_base + 2 * w_in + 1, mask=active, other=-float("inf")).to(tl.float32))
    x22 = _relu_preserve_nan(tl.load(input_ptr + input_base + 2 * w_in + 2, mask=active, other=-float("inf")).to(tl.float32))

    best = x00
    best = _take_value(x01, best)
    best = _take_value(x02, best)
    best = _take_value(x10, best)
    best = _take_value(x11, best)
    best = _take_value(x12, best)
    best = _take_value(x20, best)
    best = _take_value(x21, best)
    best = _take_value(x22, best)

    tl.store(values_ptr + linear, best, mask=active)


# 27e7b058: (T([128,192,27,27], bf16), S([3,3]), S([2,2]))
@oracle_impl(hardware="B200", point="27e7b058", KERNEL=3, BLOCK=512, num_warps=4)
# 9594e3d9: (T([128,64,55,55], bf16), S([3,3]), S([2,2]))
@oracle_impl(hardware="B200", point="9594e3d9", KERNEL=3, BLOCK=512, num_warps=4)
# 0bd9d057: (T([4,512,28,28], bf16), S([2,2]), S([2,2]))
@oracle_impl(hardware="B200", point="0bd9d057", KERNEL=2, BLOCK=512, num_warps=4)
# cb475406: (T([4,256,56,56], bf16), S([2,2]), S([2,2]))
@oracle_impl(hardware="B200", point="cb475406", KERNEL=2, BLOCK=512, num_warps=4)
# dbbbb2f7: (T([4,128,112,112], bf16), S([2,2]), S([2,2]))
@oracle_impl(hardware="B200", point="dbbbb2f7", KERNEL=2, BLOCK=512, num_warps=4)
# 484bbe51: (T([4,64,224,224], bf16), S([2,2]), S([2,2]))
@oracle_impl(hardware="B200", point="484bbe51", KERNEL=2, BLOCK=512, num_warps=4)
def oracle_forward(inputs, *, KERNEL, BLOCK, num_warps):
    arg0_1 = inputs[0]
    batch = int(arg0_1.shape[0])
    channels = int(arg0_1.shape[1])
    h_in = int(arg0_1.shape[2])
    w_in = int(arg0_1.shape[3])
    h_out = (h_in - KERNEL) // 2 + 1
    w_out = (w_in - KERNEL) // 2 + 1
    out = torch.empty_strided(
        (batch, channels, h_out, w_out),
        (channels * h_out * w_out, h_out * w_out, w_out, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    total_outputs = out.numel()

    if KERNEL == 3:
        _relu_maxpool3_values_kernel[(triton.cdiv(total_outputs, BLOCK),)](
            arg0_1,
            out,
            total_outputs=total_outputs,
            h_in=h_in,
            w_in=w_in,
            h_out=h_out,
            w_out=w_out,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
    else:
        _relu_maxpool2_values_kernel[(triton.cdiv(total_outputs, BLOCK),)](
            arg0_1,
            out,
            total_outputs=total_outputs,
            h_in=h_in,
            w_in=w_in,
            h_out=h_out,
            w_out=w_out,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
    return out
