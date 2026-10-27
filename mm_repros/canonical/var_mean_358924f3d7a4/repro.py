"""
Standalone repro captured via capture_hook.
Label: M2M100ForConditionalGeneration
Pattern hash: 358924f3d7a4
Shape hash: e6382c9e
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
    def forward(self, arg0_1: "bf16[128112, 1024]", arg1_1: "i64[64, 128]", arg2_1: "i64[64, 128]", arg3_1: "i32[64, 128]", arg4_1: "bf16[1026, 1024]", arg5_1: "bf16[1024]", arg6_1: "bf16[1024]", arg7_1: "bf16[1024, 1024]", arg8_1: "bf16[1024]", arg9_1: "bf16[1024, 1024]", arg10_1: "bf16[1024]", arg11_1: "bf16[1024, 1024]", arg12_1: "bf16[1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9):
        # No stacktrace found for following nodes
        embedding: "bf16[64, 128, 1024]" = torch.ops.aten.embedding.default(arg0_1, arg1_1, 1);  arg0_1 = arg1_1 = None
        mul: "bf16[64, 128, 1024]" = torch.ops.aten.mul.Tensor(embedding, 32.0);  embedding = None
        convert_element_type: "i32[64, 128]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.int32);  arg2_1 = None
        add: "i32[64, 128]" = torch.ops.aten.add.Tensor(convert_element_type, 0);  convert_element_type = None
        mul_1: "i32[64, 128]" = torch.ops.aten.mul.Tensor(add, arg3_1);  add = arg3_1 = None
        convert_element_type_1: "i64[64, 128]" = torch.ops.prims.convert_element_type.default(mul_1, torch.int64);  mul_1 = None
        add_1: "i64[64, 128]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1);  convert_element_type_1 = None
        view: "i64[8192]" = torch.ops.aten.view.default(add_1, [-1]);  add_1 = None
        index: "bf16[8192, 1024]" = torch.ops.aten.index.Tensor(arg4_1, [view]);  arg4_1 = view = None
        view_1: "bf16[64, 128, 1024]" = torch.ops.aten.view.default(index, _shape_param_0);  index = _shape_param_0 = None
        add_2: "bf16[64, 128, 1024]" = torch.ops.aten.add.Tensor(mul, view_1);  mul = view_1 = None
        convert_element_type_2: "f32[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_2, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_2, [2], correction = 0, keepdim = True)
        getitem: "f32[64, 128, 1]" = var_mean[0]
        getitem_1: "f32[64, 128, 1]" = var_mean[1];  var_mean = None
        sub: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_2, getitem_1);  convert_element_type_2 = getitem_1 = None
        add_3: "f32[64, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[64, 128, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        mul_2: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_3: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_2, arg5_1);  mul_2 = arg5_1 = None
        add_4: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(mul_3, arg6_1);  mul_3 = arg6_1 = None
        convert_element_type_3: "bf16[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None
        view_2: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_1);  _shape_param_1 = None
        permute: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg8_1, view_2, permute);  arg8_1 = view_2 = permute = None
        view_3: "bf16[64, 128, 1024]" = torch.ops.aten.view.default(addmm, _shape_param_2);  addmm = _shape_param_2 = None
        view_4: "bf16[64, 128, 16, 64]" = torch.ops.aten.view.default(view_3, _shape_param_3);  view_3 = _shape_param_3 = None
        permute_1: "bf16[64, 16, 128, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        view_5: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_4);  _shape_param_4 = None
        permute_2: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_1: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg10_1, view_5, permute_2);  arg10_1 = view_5 = permute_2 = None
        view_6: "bf16[64, 128, 1024]" = torch.ops.aten.view.default(addmm_1, _shape_param_5);  addmm_1 = _shape_param_5 = None
        view_7: "bf16[64, 128, 16, 64]" = torch.ops.aten.view.default(view_6, _shape_param_6);  view_6 = _shape_param_6 = None
        permute_3: "bf16[64, 16, 128, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
        view_8: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_7);  convert_element_type_3 = _shape_param_7 = None
        permute_4: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_2: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg12_1, view_8, permute_4);  arg12_1 = view_8 = permute_4 = None
        view_9: "bf16[64, 128, 1024]" = torch.ops.aten.view.default(addmm_2, _shape_param_8);  addmm_2 = _shape_param_8 = None
        view_10: "bf16[64, 128, 16, 64]" = torch.ops.aten.view.default(view_9, _shape_param_9);  view_9 = _shape_param_9 = None
        permute_5: "bf16[64, 16, 128, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        return (add_2, permute_1, permute_3, permute_5)



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
