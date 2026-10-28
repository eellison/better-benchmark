"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/GoogleFnet/full_graph_001.py
Pattern hash: fdd47407ba52
Shape hash: bf394e40
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[32000, 768]", arg1_1: "f32[16384, 32000]", arg2_1: "f32[768]", arg3_1: "f32[16384, 768]", arg4_1: "f32[32, 512, 1]", arg5_1: "f32[32, 512, 1]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        permute: "f32[768, 32000]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        permute_1: "f32[32000, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm: "f32[16384, 768]" = torch.ops.aten.mm.default(arg1_1, permute_1);  arg1_1 = permute_1 = None
        view: "f32[32, 512, 768]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view, arg2_1);  arg2_1 = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, 768)
        view_1: "f32[32, 512, 768]" = torch.ops.aten.view.default(arg3_1, _shape_param_1);  arg3_1 = _shape_param_1 = None
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_1, 0.5)
        pow_1: "f32[32, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(view_1, 3.0)
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_1, mul_3);  view_1 = mul_3 = None
        mul_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add, 0.7978845608028654);  add = None
        tanh: "f32[32, 512, 768]" = torch.ops.aten.tanh.default(mul_4);  mul_4 = None
        add_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_5: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, add_1);  mul_2 = add_1 = None
        sub: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_5, arg4_1);  mul_5 = arg4_1 = None
        mul_6: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub, arg5_1);  sub = arg5_1 = None
        mul_7: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, mul_6)
        mul_8: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view, mul_6);  view = mul_6 = None
        return (mul, mul_1, mul_7, mul_8)



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
