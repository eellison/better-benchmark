"""
Standalone repro captured via capture_hook.
Label: repros/models/torchbench/train/densenet121/full_graph_000.py
Pattern hash: 03a8d4631165
Shape hash: dfb4d847
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
    def forward(self, arg0_1: "bf16[4, 1024, 7, 7]", arg1_1: "f32[1024]", arg2_1: "f32[1024]", arg3_1: "f32[1000]", arg4_1: "f32[1000, 1024]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type: "f32[4, 1024, 7, 7]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type = None
        getitem: "f32[1, 1024, 1, 1]" = var_mean[0]
        getitem_1: "f32[1, 1024, 1, 1]" = var_mean[1];  var_mean = None
        add: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[4, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(arg0_1, getitem_1);  arg0_1 = getitem_1 = None
        mul: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        unsqueeze: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg1_1, -1);  arg1_1 = None
        unsqueeze_1: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_1: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_3: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_1: "f32[4, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_3);  mul_1 = unsqueeze_3 = None
        convert_element_type_1: "bf16[4, 1024, 7, 7]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        relu: "bf16[4, 1024, 7, 7]" = torch.ops.aten.relu.default(convert_element_type_1);  convert_element_type_1 = None
        mean: "bf16[4, 1024, 1, 1]" = torch.ops.aten.mean.dim(relu, [-1, -2], True);  relu = None
        view: "bf16[4, 1024]" = torch.ops.aten.view.default(mean, _shape_param_0);  mean = _shape_param_0 = None
        convert_element_type_2: "bf16[1000]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        convert_element_type_3: "bf16[1000, 1024]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.bfloat16);  arg4_1 = None
        permute: "bf16[1024, 1000]" = torch.ops.aten.permute.default(convert_element_type_3, [1, 0]);  convert_element_type_3 = None
        addmm: "bf16[4, 1000]" = torch.ops.aten.addmm.default(convert_element_type_2, view, permute);  convert_element_type_2 = view = permute = None
        return addmm



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
