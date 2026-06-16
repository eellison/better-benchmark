"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_infer
Pattern hash: d3e13f25b442
Shape hash: 1ae3d509
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
    def forward(self, arg0_1: "bf16[25088, 512]", arg1_1: "bf16[128, 196, 512]", arg2_1: "bf16[512]", arg3_1: "bf16[512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        view: "bf16[128, 196, 512]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        add: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(arg1_1, view);  arg1_1 = view = None
        view_1: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add, _shape_param_1);  add = _shape_param_1 = None
        convert_element_type: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [3], correction = 0, keepdim = True)
        getitem: "f32[128, 14, 14, 1]" = var_mean[0]
        getitem_1: "f32[128, 14, 14, 1]" = var_mean[1];  var_mean = None
        sub: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_1: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = arg2_1 = None
        add_2: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        convert_element_type_1: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        iota: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_3: "i64[14]" = torch.ops.aten.add.Tensor(iota, 3);  iota = None
        fmod: "i64[14]" = torch.ops.aten.fmod.Scalar(add_3, 14);  add_3 = None
        index: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(convert_element_type_1, [None, fmod]);  convert_element_type_1 = fmod = None
        iota_1: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_4: "i64[14]" = torch.ops.aten.add.Tensor(iota_1, 3);  iota_1 = None
        fmod_1: "i64[14]" = torch.ops.aten.fmod.Scalar(add_4, 14);  add_4 = None
        index_1: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index, [None, None, fmod_1]);  index = fmod_1 = None
        view_2: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_1, _shape_param_2);  index_1 = _shape_param_2 = None
        permute: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_2, [0, 1, 3, 2, 4, 5]);  view_2 = None
        clone: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_3: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        view_4: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_3, _shape_param_4);  view_3 = _shape_param_4 = None
        view_5: "bf16[25088, 512]" = torch.ops.aten.view.default(view_4, _shape_param_5);  view_4 = _shape_param_5 = None
        return (view_1, view_5)



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
