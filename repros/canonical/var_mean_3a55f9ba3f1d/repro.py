"""
Standalone repro captured via capture_hook.
Label: hf_XLNetLMHeadModel_infer
Pattern hash: 3a55f9ba3f1d
Shape hash: c4c9bec6
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
    def forward(self, arg0_1: "bf16[8192, 1024]", arg1_1: "bf16[512, 16, 1024]", arg2_1: "bf16[1024]", arg3_1: "bf16[1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        add: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view, arg1_1);  view = arg1_1 = None
        convert_element_type: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[512, 16, 1]" = var_mean[0]
        getitem_1: "f32[512, 16, 1]" = var_mean[1];  var_mean = None
        sub: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_1: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = arg2_1 = None
        add_2: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        convert_element_type_1: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        unsqueeze: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 3)
        unsqueeze_1: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 4);  unsqueeze = None
        view_1: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_1, _shape_param_1);  unsqueeze_1 = _shape_param_1 = None
        squeeze: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_1, 0);  view_1 = None
        unsqueeze_2: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 3)
        unsqueeze_3: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 4);  unsqueeze_2 = None
        view_2: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_3, _shape_param_2);  unsqueeze_3 = _shape_param_2 = None
        squeeze_1: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_2, 0);  view_2 = None
        unsqueeze_4: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 3)
        unsqueeze_5: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 4);  unsqueeze_4 = None
        view_3: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_5, _shape_param_3);  unsqueeze_5 = _shape_param_3 = None
        squeeze_2: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_3, 0);  view_3 = None
        slice_1: "bf16[512, 16, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_1, 0, -512, 9223372036854775807)
        return (convert_element_type_1, squeeze, squeeze_1, squeeze_2, slice_1)



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
