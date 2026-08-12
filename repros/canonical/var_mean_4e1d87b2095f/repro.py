"""
Standalone repro captured via capture_hook.
Label: hf_XLNetLMHeadModel_train
Pattern hash: 4e1d87b2095f
Shape hash: bc741f9d
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
    def forward(self, arg0_1: "bf16[8192, 1024]", arg1_1: "i64[99]", arg2_1: "f32[512, 16, 1024]", arg3_1: "f32[1024]", arg4_1: "f32[1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        unsqueeze: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(arg0_1, 0);  arg0_1 = None
        view: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze, _shape_param_0);  unsqueeze = _shape_param_0 = None
        permute: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view, [0, 1, 4, 2, 3]);  view = None
        view_1: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute, _shape_param_1);  permute = _shape_param_1 = None
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg1_1, 83);  arg1_1 = None
        inductor_random: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_2, inductor_lookup_seed, 'rand');  _shape_param_2 = inductor_lookup_seed = None
        convert_element_type: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type, 0.1);  convert_element_type = None
        mul: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt, view_1);  view_1 = None
        mul_1: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        add: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_1, arg2_1);  mul_1 = arg2_1 = None
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[512, 16, 1]" = var_mean[0]
        getitem_1: "f32[512, 16, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
        mul_2: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_3: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_2, arg3_1);  arg3_1 = None
        add_2: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_3, arg4_1);  mul_3 = arg4_1 = None
        convert_element_type_1: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16)
        view_2: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_3);  convert_element_type_1 = _shape_param_3 = None
        div: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None
        return (gt, mul_2, add_2, view_2, div)



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
