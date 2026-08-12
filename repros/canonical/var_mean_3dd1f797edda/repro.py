"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_train
Pattern hash: 3dd1f797edda
Shape hash: 726994b7
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
    def forward(self, arg0_1: "f32[768]", arg1_1: "bf16[8192, 768]", arg2_1: "i64[36]", arg3_1: "f32[8, 1024, 768]", arg4_1: "f32[768]", arg5_1: "f32[768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        view: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(arg1_1, _shape_param_0);  arg1_1 = _shape_param_0 = None
        add: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view, convert_element_type);  view = convert_element_type = None
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg2_1, 22);  arg2_1 = None
        inductor_random: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default(_shape_param_1, inductor_lookup_seed, 'rand');  _shape_param_1 = inductor_lookup_seed = None
        convert_element_type_1: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_1, 0.1);  convert_element_type_1 = None
        mul: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt, add);  add = None
        mul_1: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        add_1: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 1024, 1]" = var_mean[0]
        getitem_1: "f32[8, 1024, 1]" = var_mean[1];  var_mean = None
        add_2: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        mul_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2, arg4_1);  arg4_1 = None
        add_3: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_3, arg5_1);  mul_3 = arg5_1 = None
        convert_element_type_2: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16)
        view_1: "bf16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_2);  convert_element_type_2 = _shape_param_2 = None
        div: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        return (gt, mul_2, add_3, view_1, div)



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
