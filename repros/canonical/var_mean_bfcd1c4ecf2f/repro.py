"""
Standalone repro captured via capture_hook.
Label: torchbench_functorch_dp_cifar10_train
Pattern hash: bfcd1c4ecf2f
Shape hash: a488e261
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
    def forward(self, arg0_1: "bf16[128, 64, 8, 8]", arg1_1: "f32[64]", arg2_1: "f32[64]", arg3_1: "f32[128, 64, 8, 8]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        convert_element_type: "f32[128, 64, 8, 8]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        view: "f32[128, 32, 2, 64]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        var_mean = torch.ops.aten.var_mean.correction(view, [2, 3], correction = 0, keepdim = True)
        getitem: "f32[128, 32, 1, 1]" = var_mean[0]
        getitem_1: "f32[128, 32, 1, 1]" = var_mean[1];  var_mean = None
        add: "f32[128, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[128, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[128, 32, 2, 64]" = torch.ops.aten.sub.Tensor(view, getitem_1);  view = None
        mul: "f32[128, 32, 2, 64]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        view_1: "f32[128, 64, 8, 8]" = torch.ops.aten.view.default(mul, _shape_param_1);  mul = _shape_param_1 = None
        unsqueeze: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(arg1_1, 0);  arg1_1 = None
        unsqueeze_1: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        unsqueeze_2: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        mul_1: "f32[128, 64, 8, 8]" = torch.ops.aten.mul.Tensor(view_1, unsqueeze_2);  view_1 = unsqueeze_2 = None
        unsqueeze_3: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(arg2_1, 0);  arg2_1 = None
        unsqueeze_4: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
        unsqueeze_5: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        add_1: "f32[128, 64, 8, 8]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
        squeeze: "f32[128, 32]" = torch.ops.aten.squeeze.dims(getitem_1, [2, 3]);  getitem_1 = None
        squeeze_1: "f32[128, 32]" = torch.ops.aten.squeeze.dims(rsqrt, [2, 3]);  rsqrt = None
        add_2: "f32[128, 64, 8, 8]" = torch.ops.aten.add.Tensor(add_1, arg3_1);  add_1 = arg3_1 = None
        relu: "f32[128, 64, 8, 8]" = torch.ops.aten.relu.default(add_2);  add_2 = None
        convert_element_type_1: "bf16[128, 64, 8, 8]" = torch.ops.prims.convert_element_type.default(relu, torch.bfloat16)
        le: "b8[128, 64, 8, 8]" = torch.ops.aten.le.Scalar(relu, 0)
        return (squeeze, squeeze_1, relu, convert_element_type_1, le)



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
