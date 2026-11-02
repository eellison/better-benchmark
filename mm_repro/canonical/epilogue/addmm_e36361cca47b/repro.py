"""Curated MM repro. Pattern hash: e36361cca47b."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
from mm_replay import make_inputs_from_config, benchmark_repro_safe as benchmark_repro
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[4096]", arg1_1: "bf16[1000, 2880]", arg2_1: "bf16[2880, 4096]", arg3_1: "bf16[1000, 512]", arg4_1: "bf16[32]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11):
        # No stacktrace found for following nodes
        addmm: "bf16[1000, 4096]" = torch.ops.aten.addmm.default(arg0_1, arg1_1, arg2_1);  arg0_1 = arg1_1 = arg2_1 = None
        view: "bf16[1, 1000, 4096]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        view_1: "bf16[1, 1000, 64, 64]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute: "bf16[1, 64, 1000, 64]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        split = torch.ops.aten.split.Tensor(permute, 32, -1);  permute = None
        getitem: "bf16[1, 64, 1000, 32]" = split[0]
        getitem_1: "bf16[1, 64, 1000, 32]" = split[1];  split = None
        view_2: "bf16[1, 1000, 512]" = torch.ops.aten.view.default(arg3_1, _shape_param_2);  arg3_1 = _shape_param_2 = None
        view_3: "bf16[1, 1000, 8, 64]" = torch.ops.aten.view.default(view_2, _shape_param_3);  view_2 = _shape_param_3 = None
        permute_1: "bf16[1, 8, 1000, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        split_1 = torch.ops.aten.split.Tensor(permute_1, 32, -1);  permute_1 = None
        getitem_2: "bf16[1, 8, 1000, 32]" = split_1[0]
        getitem_3: "bf16[1, 8, 1000, 32]" = split_1[1];  split_1 = None
        unsqueeze: "bf16[1, 32]" = torch.ops.aten.unsqueeze.default(arg4_1, 0);  arg4_1 = None
        unsqueeze_1: "bf16[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        convert_element_type: "f32[1, 32, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_1, torch.float32);  unsqueeze_1 = None
        expand: "f32[1, 32, 1]" = torch.ops.aten.expand.default(convert_element_type, [1, -1, 1]);  convert_element_type = None
        expand_1: "f32[1, 32, 1]" = torch.ops.aten.expand.default(expand, _shape_param_4);  expand = _shape_param_4 = None
        iota: "i64[1000]" = torch.ops.prims.iota.default(1000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[1000]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None
        unsqueeze_2: "i64[1, 1000]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None
        unsqueeze_3: "i64[1, 1, 1000]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 1);  unsqueeze_2 = None
        convert_element_type_1: "f32[1, 1, 1000]" = torch.ops.prims.convert_element_type.default(unsqueeze_3, torch.float32);  unsqueeze_3 = None
        expand_2: "f32[1, 1, 1000]" = torch.ops.aten.expand.default(convert_element_type_1, _shape_param_5);  convert_element_type_1 = _shape_param_5 = None
        mul: "f32[1, 32, 1000]" = torch.ops.aten.mul.Tensor(expand_1, expand_2);  expand_1 = expand_2 = None
        permute_2: "f32[1, 1000, 32]" = torch.ops.aten.permute.default(mul, [0, 2, 1]);  mul = None
        cos: "f32[1, 1000, 32]" = torch.ops.aten.cos.default(permute_2)
        mul_1: "f32[1, 1000, 32]" = torch.ops.aten.mul.Tensor(cos, 1.3465735902799727);  cos = None
        convert_element_type_2: "bf16[1, 1000, 32]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        unsqueeze_4: "bf16[1, 1, 1000, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)
        mul_2: "bf16[1, 64, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem, unsqueeze_4)
        sin: "f32[1, 1000, 32]" = torch.ops.aten.sin.default(permute_2);  permute_2 = None
        mul_3: "f32[1, 1000, 32]" = torch.ops.aten.mul.Tensor(sin, 1.3465735902799727);  sin = None
        convert_element_type_3: "bf16[1, 1000, 32]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        unsqueeze_5: "bf16[1, 1, 1000, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)
        mul_4: "bf16[1, 64, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem_1, unsqueeze_5)
        sub: "bf16[1, 64, 1000, 32]" = torch.ops.aten.sub.Tensor(mul_2, mul_4);  mul_2 = mul_4 = None
        mul_5: "bf16[1, 64, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem_1, unsqueeze_4);  getitem_1 = None
        mul_6: "bf16[1, 64, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem, unsqueeze_5);  getitem = None
        add_1: "bf16[1, 64, 1000, 32]" = torch.ops.aten.add.Tensor(mul_5, mul_6);  mul_5 = mul_6 = None
        cat: "bf16[1, 64, 1000, 64]" = torch.ops.aten.cat.default([sub, add_1], -1);  sub = add_1 = None
        expand_3: "bf16[1, 64, 1000, 64]" = torch.ops.aten.expand.default(cat, _shape_param_6);  cat = _shape_param_6 = None
        view_4: "bf16[64, 1000, 64]" = torch.ops.aten.view.default(expand_3, _shape_param_7);  expand_3 = _shape_param_7 = None
        mul_7: "bf16[1, 8, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem_2, unsqueeze_4)
        mul_8: "bf16[1, 8, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem_3, unsqueeze_5)
        sub_1: "bf16[1, 8, 1000, 32]" = torch.ops.aten.sub.Tensor(mul_7, mul_8);  mul_7 = mul_8 = None
        mul_9: "bf16[1, 8, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem_3, unsqueeze_4);  getitem_3 = unsqueeze_4 = None
        mul_10: "bf16[1, 8, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem_2, unsqueeze_5);  getitem_2 = unsqueeze_5 = None
        add_2: "bf16[1, 8, 1000, 32]" = torch.ops.aten.add.Tensor(mul_9, mul_10);  mul_9 = mul_10 = None
        cat_1: "bf16[1, 8, 1000, 64]" = torch.ops.aten.cat.default([sub_1, add_2], -1);  sub_1 = add_2 = None
        unsqueeze_6: "bf16[1, 8, 1, 1000, 64]" = torch.ops.aten.unsqueeze.default(cat_1, 2)
        expand_4: "bf16[1, 8, 8, 1000, 64]" = torch.ops.aten.expand.default(unsqueeze_6, _shape_param_8);  unsqueeze_6 = _shape_param_8 = None
        clone: "bf16[1, 8, 8, 1000, 64]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_5: "bf16[1, 64, 1000, 64]" = torch.ops.aten.view.default(clone, _shape_param_9);  clone = _shape_param_9 = None
        permute_3: "bf16[1, 64, 64, 1000]" = torch.ops.aten.permute.default(view_5, [0, 1, 3, 2]);  view_5 = None
        expand_5: "bf16[1, 64, 64, 1000]" = torch.ops.aten.expand.default(permute_3, _shape_param_10);  permute_3 = _shape_param_10 = None
        view_6: "bf16[64, 64, 1000]" = torch.ops.aten.view.default(expand_5, _shape_param_11);  expand_5 = _shape_param_11 = None
        slice_1: "bf16[1, 8, 127, 64]" = torch.ops.aten.slice.Tensor(cat_1, 2, -127, 9223372036854775807);  cat_1 = None
        return (convert_element_type_2, convert_element_type_3, view_4, view_6, slice_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
