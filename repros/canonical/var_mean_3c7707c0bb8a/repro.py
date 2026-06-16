"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_train
Pattern hash: 3c7707c0bb8a
Shape hash: a0e6bc91
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
    def forward(self, arg0_1: "bf16[4096, 128]", arg1_1: "f32[128]", arg2_1: "f32[128]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view: "bf16[8, 512, 128]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        mul: "bf16[8, 512, 128]" = torch.ops.aten.mul.Tensor(view, 0.5)
        convert_element_type: "f32[8, 512, 128]" = torch.ops.prims.convert_element_type.default(view, torch.float32)
        pow_1: "f32[8, 512, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 3.0);  convert_element_type = None
        mul_1: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(view, mul_1);  view = mul_1 = None
        mul_2: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(add, 0.7978845608028654);  add = None
        tanh: "f32[8, 512, 128]" = torch.ops.aten.tanh.default(mul_2);  mul_2 = None
        add_1: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_3: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul, add_1);  mul = add_1 = None
        var_mean = torch.ops.aten.var_mean.correction(mul_3, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 512, 1]" = var_mean[0]
        getitem_1: "f32[8, 512, 1]" = var_mean[1];  var_mean = None
        add_2: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(mul_3, getitem_1);  mul_3 = None
        mul_4: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_5: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_4, arg1_1);  mul_4 = arg1_1 = None
        add_3: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(mul_5, arg2_1);  mul_5 = arg2_1 = None
        convert_element_type_1: "bf16[8, 512, 128]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        view_1: "bf16[4096, 128]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_1);  convert_element_type_1 = _shape_param_1 = None
        return (getitem_1, rsqrt, view_1)



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
