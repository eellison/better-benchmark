"""
Standalone repro captured via capture_hook.
Label: torchbench_tts_angular_train
Pattern hash: 08d094f67bb4
Shape hash: e9cdab09
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([64, 256], f32), T([64, 1], f32), T([64, 256], f32, stride=(12800, 1)), S([64, 256]))"

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[64, 256]", pow_2: "f32[64, 1]", select: "f32[64, 256]", _shape_param_0):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/tts_angular/model.py:75 in torch_dynamo_resume_in_forward_at_73, code: d = torch.nn.functional.normalize(d[:, -1], p=2, dim=1)
        neg_default: "f32[64, 256]" = torch.ops.aten.neg.default(tangents_1)
        clamp_min_default: "f32[64, 1]" = torch.ops.aten.clamp_min.default(pow_2, 1e-12)
        expand_default: "f32[64, 256]" = torch.ops.aten.expand.default(clamp_min_default, _shape_param_0);  clamp_min_default = _shape_param_0 = None
        div_tensor: "f32[64, 256]" = torch.ops.aten.div.Tensor(select, expand_default)
        div_tensor_1: "f32[64, 256]" = torch.ops.aten.div.Tensor(div_tensor, expand_default);  div_tensor = None
        mul_tensor: "f32[64, 256]" = torch.ops.aten.mul.Tensor(neg_default, div_tensor_1);  neg_default = div_tensor_1 = None
        div_tensor_2: "f32[64, 256]" = torch.ops.aten.div.Tensor(tangents_1, expand_default);  tangents_1 = expand_default = None
        sum_dim_int_list: "f32[64, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [1], True);  mul_tensor = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        ge_scalar: "b8[64, 1]" = torch.ops.aten.ge.Scalar(pow_2, 1e-12)
        where_self: "f32[64, 1]" = torch.ops.aten.where.self(ge_scalar, sum_dim_int_list, full_default);  ge_scalar = sum_dim_int_list = None
        div_tensor_3: "f32[64, 256]" = torch.ops.aten.div.Tensor(select, pow_2);  select = None
        eq_scalar: "b8[64, 1]" = torch.ops.aten.eq.Scalar(pow_2, 0);  pow_2 = None
        where_self_1: "f32[64, 256]" = torch.ops.aten.where.self(eq_scalar, full_default, div_tensor_3);  eq_scalar = full_default = div_tensor_3 = None
        mul_tensor_1: "f32[64, 256]" = torch.ops.aten.mul.Tensor(where_self, where_self_1);  where_self = where_self_1 = None
        add_tensor: "f32[64, 256]" = torch.ops.aten.add.Tensor(div_tensor_2, mul_tensor_1);  div_tensor_2 = mul_tensor_1 = None
        full_default_1: "f32[64, 50, 256]" = torch.ops.aten.full.default([64, 50, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_default: "f32[64, 50, 256]" = torch.ops.aten.select_scatter.default(full_default_1, add_tensor, 1, -1);  full_default_1 = add_tensor = None
        return select_scatter_default



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
