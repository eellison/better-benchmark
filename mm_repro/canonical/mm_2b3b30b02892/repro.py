"""Curated MM repro. Pattern hash: 2b3b30b02892."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[9437184]", arg1_1: "bf16[6291456]", arg2_1: "bf16[768, 768]", arg3_1: "bf16[6291456]", arg4_1: "bf16[768, 768]", arg5_1: "bf16[768, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18):
        # No stacktrace found for following nodes
        as_strided: "bf16[96, 1536, 64]" = torch.ops.aten.as_strided.default(arg0_1, _shape_param_0, _shape_param_1, 0);  arg0_1 = _shape_param_0 = _shape_param_1 = None
        constant_pad_nd: "bf16[96, 1024, 64]" = torch.ops.aten.constant_pad_nd.default(as_strided, _shape_param_2);  as_strided = _shape_param_2 = None
        view: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(constant_pad_nd, _shape_param_3);  constant_pad_nd = _shape_param_3 = None
        permute: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view, [0, 2, 1, 3]);  view = None
        permute_1: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute, [1, 0, 2, 3]);  permute = None
        clone: "bf16[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        as_strided_1: "bf16[96, 2, 512, 64]" = torch.ops.aten.as_strided.default(arg1_1, _shape_param_4, _shape_param_5, 0);  arg1_1 = _shape_param_4 = _shape_param_5 = None
        view_2: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(as_strided_1, _shape_param_6);  as_strided_1 = _shape_param_6 = None
        view_3: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(view_2, _shape_param_7);  view_2 = _shape_param_7 = None
        permute_2: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        permute_3: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_2, [1, 0, 2, 3]);  permute_2 = None
        view_4: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(permute_3, _shape_param_8);  permute_3 = _shape_param_8 = None
        div: "bf16[1024, 8, 768]" = torch.ops.aten.div.Tensor(view_4, 8.0);  view_4 = None
        view_5: "bf16[8192, 768]" = torch.ops.aten.view.default(clone, _shape_param_9);  clone = _shape_param_9 = None
        mm: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_5, arg2_1);  view_5 = arg2_1 = None
        view_6: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm, _shape_param_10);  mm = _shape_param_10 = None
        convert_element_type: "f32[1024, 8, 768]" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None
        as_strided_2: "bf16[96, 2, 512, 64]" = torch.ops.aten.as_strided.default(arg3_1, _shape_param_11, _shape_param_12, 0);  arg3_1 = _shape_param_11 = _shape_param_12 = None
        view_7: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(as_strided_2, _shape_param_13);  as_strided_2 = _shape_param_13 = None
        view_8: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(view_7, _shape_param_14);  view_7 = _shape_param_14 = None
        permute_4: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        permute_5: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_4, [1, 0, 2, 3]);  permute_4 = None
        view_10: "bf16[8192, 768]" = torch.ops.aten.view.default(permute_5, _shape_param_15);  permute_5 = _shape_param_15 = None
        mm_1: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_10, arg4_1);  view_10 = arg4_1 = None
        view_11: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_1, _shape_param_16);  mm_1 = _shape_param_16 = None
        convert_element_type_1: "f32[1024, 8, 768]" = torch.ops.prims.convert_element_type.default(view_11, torch.float32);  view_11 = None
        add: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(convert_element_type, convert_element_type_1);  convert_element_type = convert_element_type_1 = None
        view_12: "bf16[8192, 768]" = torch.ops.aten.view.default(div, _shape_param_17);  div = _shape_param_17 = None
        mm_2: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_12, arg5_1);  view_12 = arg5_1 = None
        view_13: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_2, _shape_param_18);  mm_2 = _shape_param_18 = None
        convert_element_type_2: "f32[1024, 8, 768]" = torch.ops.prims.convert_element_type.default(view_13, torch.float32);  view_13 = None
        add_1: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(add, convert_element_type_2);  add = convert_element_type_2 = None
        permute_6: "f32[8, 1024, 768]" = torch.ops.aten.permute.default(add_1, [1, 0, 2]);  add_1 = None
        return permute_6

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
