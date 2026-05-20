"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-2-5-linux.aws.a100_graph5
Pattern hash: d8ac9dba20da
Shape hash: f01acef8
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([], f32), T([], f32), T([1, 512], i64, max=512), T([512, 32000], f32), T([512, 1], f32), T([512, 1], f32), T([32000, 768], f32), S([1, 32000]), S([512, 32000]))"

class Repro(torch.nn.Module):
    def forward(self, arg166_1: "f32[]", arg140_1: "f32[]", arg56_1: "i64[1, 512]", arg137_1: "f32[512, 32000]", arg138_1: "f32[512, 1]", arg139_1: "f32[512, 1]", arg3_1: "f32[32000, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(arg166_1, arg140_1);  arg166_1 = arg140_1 = None
        view_default: "i64[512]" = torch.ops.aten.view.default(arg56_1, [-1]);  arg56_1 = None
        unsqueeze_default: "i64[512, 1]" = torch.ops.aten.unsqueeze.default(view_default, 1);  view_default = None
        ne_scalar: "b8[512, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_default, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[512, 1]" = torch.ops.aten.where.self(ne_scalar, unsqueeze_default, full_default);  unsqueeze_default = full_default = None
        iota_default: "i64[32000]" = torch.ops.prims.iota.default(32000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default_1: "i64[1, 32000]" = torch.ops.aten.view.default(iota_default, _shape_param_0);  iota_default = _shape_param_0 = None
        expand_default: "i64[512, 32000]" = torch.ops.aten.expand.default(where_self, _shape_param_1);  where_self = _shape_param_1 = None
        eq_tensor: "b8[512, 32000]" = torch.ops.aten.eq.Tensor(expand_default, view_default_1);  expand_default = view_default_1 = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where_self_1: "f32[512, 32000]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_2: "f32[512, 1]" = torch.ops.aten.where.self(ne_scalar, div_tensor, full_default_1);  ne_scalar = div_tensor = full_default_1 = None
        mul_tensor: "f32[512, 32000]" = torch.ops.aten.mul.Tensor(where_self_1, where_self_2);  where_self_1 = where_self_2 = None
        sub_tensor: "f32[512, 32000]" = torch.ops.aten.sub.Tensor(arg137_1, arg138_1);  arg137_1 = arg138_1 = None
        sub_tensor_1: "f32[512, 32000]" = torch.ops.aten.sub.Tensor(sub_tensor, arg139_1);  sub_tensor = arg139_1 = None
        exp_default: "f32[512, 32000]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        sum_dim_int_list: "f32[512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [1], True)
        mul_tensor_1: "f32[512, 32000]" = torch.ops.aten.mul.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        sub_tensor_2: "f32[512, 32000]" = torch.ops.aten.sub.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        permute_default: "f32[768, 32000]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        permute_default_1: "f32[32000, 768]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (sub_tensor_2, permute_default_1)


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
