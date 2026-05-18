"""
Standalone repro captured via capture_hook.
Label: tritonbench_kl_div_4096x32768
Pattern hash: 4e221e32dd02
Shape hash: f2328034
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[4096, 32768]", arg0_1: "f32[4096, 32768]"):
        # File: /tmp/scratch_space/better_benchmark/capture_tritonbench_ops.py:142 in kl_div_fn, code: return F.kl_div(log_probs, target_probs, reduction='batchmean')
        isnan_default: "b8[4096, 32768]" = torch.ops.aten.isnan.default(arg1_1)
        full_default: "f32[]" = torch.ops.aten.full.default([], nan, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        eq_scalar: "b8[4096, 32768]" = torch.ops.aten.eq.Scalar(arg1_1, 0)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        log_default: "f32[4096, 32768]" = torch.ops.aten.log.default(arg1_1)
        mul_tensor: "f32[4096, 32768]" = torch.ops.aten.mul.Tensor(arg1_1, log_default);  log_default = None
        where_self: "f32[4096, 32768]" = torch.ops.aten.where.self(eq_scalar, full_default_1, mul_tensor);  eq_scalar = full_default_1 = mul_tensor = None
        where_self_1: "f32[4096, 32768]" = torch.ops.aten.where.self(isnan_default, full_default, where_self);  isnan_default = full_default = where_self = None
        mul_tensor_1: "f32[4096, 32768]" = torch.ops.aten.mul.Tensor(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        sub_tensor: "f32[4096, 32768]" = torch.ops.aten.sub.Tensor(where_self_1, mul_tensor_1);  where_self_1 = mul_tensor_1 = None
        sum_default: "f32[]" = torch.ops.aten.sum.default(sub_tensor);  sub_tensor = None
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(sum_default, 4096);  sum_default = None
        return div_tensor


def _default_make_inputs():
    return [
    torch.randn([4096, 32768], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 32768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
