"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/GoogleFnet/full_graph_001.py
Pattern hash: 8908ead50c07
Shape hash: 6641f670
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
    def forward(self, arg0_1: "f32[768, 3072]", arg1_1: "f32[16384, 768]", arg2_1: "f32[16384, 3072]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        permute: "f32[3072, 768]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        permute_1: "f32[768, 3072]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm: "f32[16384, 3072]" = torch.ops.aten.mm.default(arg1_1, permute_1);  arg1_1 = permute_1 = None
        view: "f32[32, 512, 3072]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        view_1: "f32[32, 512, 3072]" = torch.ops.aten.view.default(arg2_1, _shape_param_1);  arg2_1 = _shape_param_1 = None
        mul: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_1, 0.5)
        mul_1: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view, mul);  mul = None
        pow_1: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_1, 3.0)
        mul_2: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_1, mul_2);  mul_2 = None
        mul_3: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add, 0.7978845608028654);  add = None
        tanh: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_3);  mul_3 = None
        add_1: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0)
        mul_4: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view, add_1);  view = add_1 = None
        mul_5: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub: "f32[32, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_5);  mul_5 = None
        mul_6: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_1, sub);  mul_1 = sub = None
        mul_7: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_6, 0.7978845608028654);  mul_6 = None
        mul_8: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_7, 0.044715)
        pow_2: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_1, 2.0);  view_1 = None
        mul_9: "f32[32, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_2, 3.0);  pow_2 = None
        mul_10: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        add_2: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_7, mul_10);  mul_7 = mul_10 = None
        mul_11: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_4, 0.5);  mul_4 = None
        add_3: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(add_2, mul_11);  add_2 = mul_11 = None
        view_2: "f32[16384, 3072]" = torch.ops.aten.view.default(add_3, _shape_param_2);  add_3 = _shape_param_2 = None
        permute_2: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_2, [1, 0])
        return (view_2, permute_2)



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
