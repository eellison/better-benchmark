"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_train
Pattern hash: f7f800689be3
Shape hash: 93219f63
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
    def forward(self, arg0_1: "f32[768, 128, 3, 3]", arg1_1: "f32[768, 1, 1, 1]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        clone: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(arg0_1, memory_format = torch.contiguous_format);  arg0_1 = None
        view: "f32[1, 768, 1152]" = torch.ops.aten.view.default(clone, _shape_param_0);  clone = _shape_param_0 = None
        mul: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg1_1, 0.02946278254943948);  arg1_1 = None
        view_1: "f32[768]" = torch.ops.aten.view.default(mul, [-1]);  mul = None
        var_mean = torch.ops.aten.var_mean.correction(view, [0, 2], correction = 0, keepdim = True)
        getitem: "f32[1, 768, 1]" = var_mean[0]
        getitem_1: "f32[1, 768, 1]" = var_mean[1];  var_mean = None
        add: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(view, getitem_1);  view = None
        mul_1: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2]);  getitem_1 = None
        squeeze_1: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2]);  rsqrt = None
        unsqueeze: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1);  view_1 = None
        mul_2: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze);  mul_1 = unsqueeze = None
        view_2: "f32[768, 128, 3, 3]" = torch.ops.aten.view.default(mul_2, _shape_param_1);  mul_2 = _shape_param_1 = None
        convert_element_type: "bf16[768, 128, 3, 3]" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        unsqueeze_1: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_2: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None
        return (squeeze_1, convert_element_type, unsqueeze_2)



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
