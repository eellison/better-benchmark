"""Curated MM repro. Pattern hash: 0f691d964688."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[1, 1000, 1]", arg1_1: "f32[1, 1000, 2880]", arg2_1: "bf16[2880]", arg3_1: "bf16[2880, 512]", arg4_1: "bf16[512]", arg5_1: "f32[1, 1000, 32]", arg6_1: "f32[1, 1000, 32]", arg7_1: "bf16[2880, 512]", arg8_1: "bf16[512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9):
        # No stacktrace found for following nodes
        add: "f32[1, 1000, 1]" = torch.ops.aten.add.Tensor(arg0_1, 1e-05);  arg0_1 = None
        rsqrt: "f32[1, 1000, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[1, 1000, 2880]" = torch.ops.aten.mul.Tensor(arg1_1, rsqrt);  arg1_1 = rsqrt = None
        mul_1: "f32[1, 1000, 2880]" = torch.ops.aten.mul.Tensor(arg2_1, mul);  arg2_1 = mul = None
        convert_element_type: "bf16[1, 1000, 2880]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        view: "bf16[1000, 2880]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  _shape_param_0 = None
        addmm: "bf16[1000, 512]" = torch.ops.aten.addmm.default(arg4_1, view, arg3_1);  arg4_1 = view = arg3_1 = None
        view_2: "bf16[1, 1000, 8, 64]" = torch.ops.aten.view.default(addmm, _shape_param_1);  addmm = _shape_param_1 = None
        permute_1: "bf16[1, 8, 1000, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        split = torch.ops.aten.split.Tensor(permute_1, 32, -1);  permute_1 = None
        getitem: "bf16[1, 8, 1000, 32]" = split[0]
        getitem_1: "bf16[1, 8, 1000, 32]" = split[1];  split = None
        mul_2: "f32[1, 1000, 32]" = torch.ops.aten.mul.Tensor(arg5_1, 1.3465735902799727);  arg5_1 = None
        convert_element_type_1: "bf16[1, 1000, 32]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        unsqueeze: "bf16[1, 1, 1000, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1);  convert_element_type_1 = None
        mul_3: "f32[1, 1000, 32]" = torch.ops.aten.mul.Tensor(arg6_1, 1.3465735902799727);  arg6_1 = None
        convert_element_type_2: "bf16[1, 1000, 32]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        unsqueeze_1: "bf16[1, 1, 1000, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1);  convert_element_type_2 = None
        mul_4: "bf16[1, 8, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem, unsqueeze)
        mul_5: "bf16[1, 8, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem_1, unsqueeze_1)
        sub: "bf16[1, 8, 1000, 32]" = torch.ops.aten.sub.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        mul_6: "bf16[1, 8, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem_1, unsqueeze);  getitem_1 = unsqueeze = None
        mul_7: "bf16[1, 8, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem, unsqueeze_1);  getitem = unsqueeze_1 = None
        add_1: "bf16[1, 8, 1000, 32]" = torch.ops.aten.add.Tensor(mul_6, mul_7);  mul_6 = mul_7 = None
        cat: "bf16[1, 8, 1000, 64]" = torch.ops.aten.cat.default([sub, add_1], -1);  sub = add_1 = None
        unsqueeze_2: "bf16[1, 8, 1, 1000, 64]" = torch.ops.aten.view.default(cat, _shape_param_2);  _shape_param_2 = None
        expand: "bf16[1, 8, 8, 1000, 64]" = torch.ops.aten.expand.default(unsqueeze_2, _shape_param_3);  unsqueeze_2 = _shape_param_3 = None
        clone: "bf16[1, 8, 8, 1000, 64]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_3: "bf16[1, 64, 1000, 64]" = torch.ops.aten.view.default(clone, _shape_param_4);  clone = _shape_param_4 = None
        permute_2: "bf16[1, 64, 64, 1000]" = torch.ops.aten.permute.default(view_3, [0, 1, 3, 2]);  view_3 = None
        view_4: "bf16[64, 64, 1000]" = torch.ops.aten.view.default(permute_2, _shape_param_5);  permute_2 = _shape_param_5 = None
        view_5: "bf16[1000, 2880]" = torch.ops.aten.view.default(convert_element_type, _shape_param_6);  convert_element_type = _shape_param_6 = None
        addmm_1: "bf16[1000, 512]" = torch.ops.aten.addmm.default(arg8_1, view_5, arg7_1);  arg8_1 = view_5 = arg7_1 = None
        view_7: "bf16[1, 1000, 8, 64]" = torch.ops.aten.view.default(addmm_1, _shape_param_7);  addmm_1 = _shape_param_7 = None
        permute_4: "bf16[1, 8, 1000, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
        unsqueeze_3: "bf16[1, 8, 1, 1000, 64]" = torch.ops.aten.unsqueeze.default(permute_4, 2)
        expand_2: "bf16[1, 8, 8, 1000, 64]" = torch.ops.aten.expand.default(unsqueeze_3, _shape_param_8);  unsqueeze_3 = _shape_param_8 = None
        clone_1: "bf16[1, 8, 8, 1000, 64]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_9: "bf16[64, 1000, 64]" = torch.ops.aten.view.default(clone_1, _shape_param_9);  clone_1 = _shape_param_9 = None
        slice_1: "bf16[1, 8, 127, 64]" = torch.ops.aten.slice.Tensor(permute_4, 2, -127, 9223372036854775807);  permute_4 = None
        slice_2: "bf16[1, 8, 127, 64]" = torch.ops.aten.slice.Tensor(cat, 2, -127, 9223372036854775807);  cat = None
        return (view_4, view_9, slice_1, slice_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
