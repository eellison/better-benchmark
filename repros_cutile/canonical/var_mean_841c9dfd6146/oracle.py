"""cuTile port of var_mean_841c9dfd6146: Visformer BN-training add + BN scope.

The reference computes:
  add = arg0.float() + arg1.float()  (bf16 arg1 -> fp32)
  var_mean(add, [0,2,3], correction=0)
  invstd = rsqrt(var + 1e-5)
  centered = add - mean, affine = ((add - mean) * invstd * weight + bias).to(bf16)
  running_mean/var mutable copy_
  return (add, invstd_1d, affine_bf16, centered, running_mean, running_var).

Plan: cuTile kernel does the fp32 add producer. Torch handles the per-channel
reduction and running-stat updates. A second cuTile kernel performs the
affine+centered epilogue using per-channel broadcast tensors materialised in
channels-last layout.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0001594642002871


@ct.kernel
def _add_kernel(
    x0_ptr,        # f32 [numel]
    x1_ptr,        # bf16 [numel]
    add_out_ptr,   # f32 [numel]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x0 = ct.load(x0_ptr, index=(pid,), shape=(BLOCK,))
    x1 = ct.load(x1_ptr, index=(pid,), shape=(BLOCK,))
    added = x0 + ct.astype(x1, ct.float32)
    ct.store(add_out_ptr, index=(pid,), tile=added)


@ct.kernel
def _affine_centered_kernel(
    add_ptr,        # f32 [numel]
    mean_ptr,       # f32 [numel] (broadcast)
    invstd_ptr,     # f32 [numel]
    weight_ptr,     # f32 [numel]
    bias_ptr,       # f32 [numel]
    affine_ptr,     # bf16 [numel]
    centered_ptr,   # f32 [numel]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    added = ct.load(add_ptr, index=(pid,), shape=(BLOCK,))
    mean = ct.load(mean_ptr, index=(pid,), shape=(BLOCK,))
    invstd = ct.load(invstd_ptr, index=(pid,), shape=(BLOCK,))
    weight = ct.load(weight_ptr, index=(pid,), shape=(BLOCK,))
    bias = ct.load(bias_ptr, index=(pid,), shape=(BLOCK,))

    centered = added - mean
    normalized = centered * invstd
    affine = normalized * weight + bias

    ct.store(centered_ptr, index=(pid,), tile=centered)
    ct.store(affine_ptr, index=(pid,), tile=ct.astype(affine, ct.bfloat16))


def _flat(t):
    """View a dense tensor as 1D interpreting physical memory in-place."""
    return t.as_strided((t.numel(),), (1,))


def _bc_like(vec_c, ref):
    """Broadcast [C] into ref's shape/stride (channels-last)."""
    n, c, h, w = ref.shape
    out = torch.empty_strided(
        (n, c, h, w),
        ref.stride(),
        device=vec_c.device,
        dtype=torch.float32,
    )
    out.copy_(vec_c.view(1, c, 1, 1).expand(n, c, h, w).to(torch.float32))
    return out


@oracle_impl(hardware="B200", point="d9d8b8eb", BLOCK=1024)
@oracle_impl(hardware="B200", point="cf9ba6ac", BLOCK=1024)
@oracle_impl(hardware="B200", point="e3b6f6ad", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    x0, x1, running_mean, running_var, weight, bias = inputs
    device = x0.device

    add_out = torch.empty_strided(
        tuple(x0.shape),
        tuple(x0.stride()),
        device=device,
        dtype=torch.float32,
    )
    affine_out = torch.empty_strided(
        tuple(x0.shape),
        tuple(x0.stride()),
        device=device,
        dtype=torch.bfloat16,
    )
    centered_out = torch.empty_strided(
        tuple(x0.shape),
        tuple(x0.stride()),
        device=device,
        dtype=torch.float32,
    )
    numel = int(x0.numel())
    assert numel % BLOCK == 0, "numel must be divisible by BLOCK for OOB safety"

    stream = torch.cuda.current_stream()

    # Kernel 1: fp32 add producer.
    ct.launch(
        stream,
        (ct.cdiv(numel, BLOCK), 1, 1),
        _add_kernel,
        (_flat(x0), _flat(x1), _flat(add_out), BLOCK),
    )

    # Torch: population var_mean over [0, 2, 3].
    var, mean = torch.var_mean(add_out, dim=[0, 2, 3], correction=0, keepdim=False)
    invstd = torch.rsqrt(var + EPS)

    # Running-stat update (mutable aliases).
    new_mean = running_mean * (1.0 - MOMENTUM) + mean * MOMENTUM
    new_var = (
        running_var * (1.0 - MOMENTUM)
        + var * RUNNING_VAR_CORRECTION * MOMENTUM
    )
    torch.ops.aten.copy_(running_mean, new_mean)
    torch.ops.aten.copy_(running_var, new_var)

    # Broadcast per-channel tensors to (N,C,H,W) matching x0's channels-last layout.
    mean_bc = _bc_like(mean, x0)
    invstd_bc = _bc_like(invstd, x0)
    weight_bc = _bc_like(weight, x0)
    bias_bc = _bc_like(bias, x0)

    # Kernel 2: affine + centered epilogue.
    ct.launch(
        stream,
        (ct.cdiv(numel, BLOCK), 1, 1),
        _affine_centered_kernel,
        (
            _flat(add_out),
            _flat(mean_bc),
            _flat(invstd_bc),
            _flat(weight_bc),
            _flat(bias_bc),
            _flat(affine_out),
            _flat(centered_out),
            BLOCK,
        ),
    )

    return add_out, invstd, affine_out, centered_out, running_mean, running_var
