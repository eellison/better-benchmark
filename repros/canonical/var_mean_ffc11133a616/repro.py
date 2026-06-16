"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_infer
Pattern hash: ffc11133a616
Shape hash: a24dc267
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
    def forward(self, arg0_1: "bf16[3072, 1536, 1, 1]", arg1_1: "bf16[3072, 1, 1, 1]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view: "bf16[1, 3072, 1536]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[1, 3072, 1536]" = torch.ops.prims.convert_element_type.default(view, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [0, 2], correction = 0, keepdim = True);  convert_element_type = None
        getitem: "f32[1, 3072, 1]" = var_mean[0]
        getitem_1: "f32[1, 3072, 1]" = var_mean[1];  var_mean = None
        sub: "f32[1, 3072, 1536]" = torch.ops.aten.sub.Tensor(view, getitem_1);  view = getitem_1 = None
        add: "f32[1, 3072, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 3072, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[1, 3072, 1536]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "bf16[3072, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg1_1, 0.02551551815399144);  arg1_1 = None
        view_1: "bf16[3072]" = torch.ops.aten.view.default(mul_1, [-1]);  mul_1 = None
        unsqueeze: "bf16[3072, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1);  view_1 = None
        mul_2: "f32[1, 3072, 1536]" = torch.ops.aten.mul.Tensor(mul, unsqueeze);  mul = unsqueeze = None
        convert_element_type_1: "bf16[1, 3072, 1536]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        view_2: "bf16[3072, 1536, 1, 1]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_1);  convert_element_type_1 = _shape_param_1 = None
        return view_2



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
