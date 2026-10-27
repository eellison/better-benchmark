"""
Standalone repro captured via capture_hook.
Label: whisper-tiny
Pattern hash: c087bfcc23c0
Shape hash: 229eb5ed
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
    def forward(self, arg0_1: "bf16[1, 384, 1500]", arg1_1: "bf16[1500, 384]", arg2_1: "bf16[384]", arg3_1: "bf16[384]", arg4_1: "bf16[384, 384]", arg5_1: "bf16[384]", arg6_1: "bf16[384, 384]", arg7_1: "bf16[384, 384]", arg8_1: "bf16[384]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # No stacktrace found for following nodes
        convert_element_type: "f32[1, 384, 1500]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        mul: "f32[1, 384, 1500]" = torch.ops.aten.mul.Tensor(convert_element_type, 0.5)
        mul_1: "f32[1, 384, 1500]" = torch.ops.aten.mul.Tensor(convert_element_type, 0.7071067811865476);  convert_element_type = None
        erf: "f32[1, 384, 1500]" = torch.ops.aten.erf.default(mul_1);  mul_1 = None
        add: "f32[1, 384, 1500]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_2: "f32[1, 384, 1500]" = torch.ops.aten.mul.Tensor(mul, add);  mul = add = None
        convert_element_type_1: "bf16[1, 384, 1500]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        permute: "bf16[1, 1500, 384]" = torch.ops.aten.permute.default(convert_element_type_1, [0, 2, 1]);  convert_element_type_1 = None
        iota: "i64[1500]" = torch.ops.prims.iota.default(1500, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        embedding: "bf16[1500, 384]" = torch.ops.aten.embedding.default(arg1_1, iota);  arg1_1 = iota = None
        add_1: "bf16[1, 1500, 384]" = torch.ops.aten.add.Tensor(permute, embedding);  permute = embedding = None
        clone: "bf16[1, 1500, 384]" = torch.ops.aten.clone.default(add_1, memory_format = torch.contiguous_format)
        convert_element_type_2: "f32[1, 1500, 384]" = torch.ops.prims.convert_element_type.default(clone, torch.float32);  clone = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_2, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 1500, 1]" = var_mean[0]
        getitem_1: "f32[1, 1500, 1]" = var_mean[1];  var_mean = None
        sub: "f32[1, 1500, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_2, getitem_1);  convert_element_type_2 = getitem_1 = None
        add_2: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_3: "f32[1, 1500, 384]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_4: "f32[1, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_3, arg2_1);  mul_3 = arg2_1 = None
        add_3: "f32[1, 1500, 384]" = torch.ops.aten.add.Tensor(mul_4, arg3_1);  mul_4 = arg3_1 = None
        convert_element_type_3: "bf16[1, 1500, 384]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        view: "bf16[1500, 384]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_0);  _shape_param_0 = None
        permute_1: "bf16[384, 384]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        addmm: "bf16[1500, 384]" = torch.ops.aten.addmm.default(arg5_1, view, permute_1);  arg5_1 = view = permute_1 = None
        view_1: "bf16[1, 1500, 384]" = torch.ops.aten.view.default(addmm, _shape_param_1);  addmm = _shape_param_1 = None
        mul_5: "bf16[1, 1500, 384]" = torch.ops.aten.mul.Tensor(view_1, 0.125);  view_1 = None
        view_2: "bf16[1, 1500, 6, 64]" = torch.ops.aten.view.default(mul_5, _shape_param_2);  mul_5 = _shape_param_2 = None
        permute_2: "bf16[1, 6, 1500, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        clone_1: "bf16[1, 6, 1500, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_3: "bf16[1500, 384]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_3);  _shape_param_3 = None
        permute_3: "bf16[384, 384]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        mm: "bf16[1500, 384]" = torch.ops.aten.mm.default(view_3, permute_3);  view_3 = permute_3 = None
        view_4: "bf16[1, 1500, 384]" = torch.ops.aten.view.default(mm, _shape_param_4);  mm = _shape_param_4 = None
        view_5: "bf16[1, 1500, 6, 64]" = torch.ops.aten.view.default(view_4, _shape_param_5);  view_4 = _shape_param_5 = None
        permute_4: "bf16[1, 6, 1500, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        clone_2: "bf16[1, 6, 1500, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        view_6: "bf16[1500, 384]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_6);  convert_element_type_3 = _shape_param_6 = None
        permute_5: "bf16[384, 384]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_1: "bf16[1500, 384]" = torch.ops.aten.addmm.default(arg8_1, view_6, permute_5);  arg8_1 = view_6 = permute_5 = None
        view_7: "bf16[1, 1500, 384]" = torch.ops.aten.view.default(addmm_1, _shape_param_7);  addmm_1 = _shape_param_7 = None
        view_8: "bf16[1, 1500, 6, 64]" = torch.ops.aten.view.default(view_7, _shape_param_8);  view_7 = _shape_param_8 = None
        permute_6: "bf16[1, 6, 1500, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        clone_3: "bf16[1, 6, 1500, 64]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
        return (add_1, clone_1, clone_2, clone_3)



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
