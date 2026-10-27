"""
Standalone repro captured via capture_hook.
Label: mobilevit_s
Pattern hash: c83a6ea39f3c
Shape hash: 909a5863
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
    def forward(self, arg0_1: "bf16[128, 192, 16, 16]", arg1_1: "bf16[192]", arg2_1: "bf16[192]", arg3_1: "bf16[576, 192]", arg4_1: "bf16[576]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        clone: "bf16[128, 192, 16, 16]" = torch.ops.aten.clone.default(arg0_1, memory_format = torch.contiguous_format);  arg0_1 = None
        view: "bf16[196608, 2, 8, 2]" = torch.ops.aten.view.default(clone, _shape_param_0);  clone = _shape_param_0 = None
        permute: "bf16[196608, 8, 2, 2]" = torch.ops.aten.permute.default(view, [0, 2, 1, 3]);  view = None
        clone_1: "bf16[196608, 8, 2, 2]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_1: "bf16[128, 192, 64, 4]" = torch.ops.aten.view.default(clone_1, _shape_param_1);  clone_1 = _shape_param_1 = None
        permute_1: "bf16[128, 4, 64, 192]" = torch.ops.aten.permute.default(view_1, [0, 3, 2, 1]);  view_1 = None
        clone_2: "bf16[128, 4, 64, 192]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_2: "bf16[512, 64, 192]" = torch.ops.aten.view.default(clone_2, _shape_param_2);  clone_2 = _shape_param_2 = None
        convert_element_type: "f32[512, 64, 192]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[512, 64, 1]" = var_mean[0]
        getitem_1: "f32[512, 64, 1]" = var_mean[1];  var_mean = None
        sub: "f32[512, 64, 192]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add: "f32[512, 64, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[512, 64, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul, arg1_1);  mul = arg1_1 = None
        add_1: "f32[512, 64, 192]" = torch.ops.aten.add.Tensor(mul_1, arg2_1);  mul_1 = arg2_1 = None
        convert_element_type_1: "bf16[512, 64, 192]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        view_3: "bf16[32768, 192]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_3);  convert_element_type_1 = _shape_param_3 = None
        permute_2: "bf16[192, 576]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        addmm: "bf16[32768, 576]" = torch.ops.aten.addmm.default(arg4_1, view_3, permute_2);  arg4_1 = view_3 = permute_2 = None
        view_4: "bf16[512, 64, 576]" = torch.ops.aten.view.default(addmm, _shape_param_4);  addmm = _shape_param_4 = None
        view_5: "bf16[512, 64, 3, 4, 48]" = torch.ops.aten.view.default(view_4, _shape_param_5);  view_4 = _shape_param_5 = None
        permute_3: "bf16[3, 512, 4, 64, 48]" = torch.ops.aten.permute.default(view_5, [2, 0, 3, 1, 4]);  view_5 = None
        unbind = torch.ops.aten.unbind.int(permute_3);  permute_3 = None
        getitem_2: "bf16[512, 4, 64, 48]" = unbind[0]
        getitem_3: "bf16[512, 4, 64, 48]" = unbind[1]
        getitem_4: "bf16[512, 4, 64, 48]" = unbind[2];  unbind = None
        return (view_2, getitem_2, getitem_3, getitem_4)



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
