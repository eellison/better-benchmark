"""
Standalone repro captured via capture_hook.
Label: torchbench_functorch_dp_cifar10_infer
Pattern hash: e56fc242c590
Shape hash: 8e339e83
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
    def forward(self, arg0_1: "bf16[128, 512, 1, 1]", arg1_1: "bf16[128, 512, 1, 1]", arg2_1: "bf16[512]", arg3_1: "bf16[512]", arg4_1: "bf16[512]", arg5_1: "bf16[512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        convert_element_type: "f32[128, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        view: "f32[128, 32, 16, 1]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        var_mean = torch.ops.aten.var_mean.correction(view, [2, 3], correction = 0, keepdim = True)
        getitem: "f32[128, 32, 1, 1]" = var_mean[0]
        getitem_1: "f32[128, 32, 1, 1]" = var_mean[1];  var_mean = None
        convert_element_type_1: "f32[128, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        view_1: "f32[128, 32, 16, 1]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_1);  convert_element_type_1 = _shape_param_1 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(view_1, [2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[128, 32, 1, 1]" = var_mean_1[0]
        getitem_3: "f32[128, 32, 1, 1]" = var_mean_1[1];  var_mean_1 = None
        sub: "f32[128, 32, 16, 1]" = torch.ops.aten.sub.Tensor(view, getitem_1);  view = getitem_1 = None
        add: "f32[128, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[128, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[128, 32, 16, 1]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        view_2: "f32[128, 512, 1, 1]" = torch.ops.aten.view.default(mul, _shape_param_2);  mul = _shape_param_2 = None
        unsqueeze: "bf16[1, 512]" = torch.ops.aten.unsqueeze.default(arg2_1, 0);  arg2_1 = None
        unsqueeze_1: "bf16[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        unsqueeze_2: "bf16[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        mul_1: "f32[128, 512, 1, 1]" = torch.ops.aten.mul.Tensor(view_2, unsqueeze_2);  view_2 = unsqueeze_2 = None
        unsqueeze_3: "bf16[1, 512]" = torch.ops.aten.unsqueeze.default(arg3_1, 0);  arg3_1 = None
        unsqueeze_4: "bf16[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
        unsqueeze_5: "bf16[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        add_1: "f32[128, 512, 1, 1]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
        convert_element_type_2: "bf16[128, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        sub_1: "f32[128, 32, 16, 1]" = torch.ops.aten.sub.Tensor(view_1, getitem_3);  view_1 = getitem_3 = None
        add_2: "f32[128, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[128, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_2: "f32[128, 32, 16, 1]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        view_3: "f32[128, 512, 1, 1]" = torch.ops.aten.view.default(mul_2, _shape_param_3);  mul_2 = _shape_param_3 = None
        unsqueeze_6: "bf16[1, 512]" = torch.ops.aten.unsqueeze.default(arg4_1, 0);  arg4_1 = None
        unsqueeze_7: "bf16[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 2);  unsqueeze_6 = None
        unsqueeze_8: "bf16[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
        mul_3: "f32[128, 512, 1, 1]" = torch.ops.aten.mul.Tensor(view_3, unsqueeze_8);  view_3 = unsqueeze_8 = None
        unsqueeze_9: "bf16[1, 512]" = torch.ops.aten.unsqueeze.default(arg5_1, 0);  arg5_1 = None
        unsqueeze_10: "bf16[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 2);  unsqueeze_9 = None
        unsqueeze_11: "bf16[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 3);  unsqueeze_10 = None
        add_3: "f32[128, 512, 1, 1]" = torch.ops.aten.add.Tensor(mul_3, unsqueeze_11);  mul_3 = unsqueeze_11 = None
        convert_element_type_3: "bf16[128, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        add_4: "bf16[128, 512, 1, 1]" = torch.ops.aten.add.Tensor(convert_element_type_2, convert_element_type_3);  convert_element_type_2 = convert_element_type_3 = None
        relu: "bf16[128, 512, 1, 1]" = torch.ops.aten.relu.default(add_4);  add_4 = None
        return relu



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
