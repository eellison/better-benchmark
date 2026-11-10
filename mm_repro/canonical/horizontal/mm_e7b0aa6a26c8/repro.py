"""Curated MM repro. Pattern hash: e7b0aa6a26c8."""
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
    def forward(self, arg0_1: "bf16[1024, 8, 768]", arg1_1: "bf16[768, 768]", arg2_1: "bf16[768]", arg3_1: "bf16[768, 768]", arg4_1: "bf16[768]", arg5_1: "bf16[768, 768]", arg6_1: "bf16[768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23):
        # No stacktrace found for following nodes
        clone: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(arg0_1, memory_format = torch.contiguous_format);  arg0_1 = None
        view: "bf16[8192, 768]" = torch.ops.aten.view.default(clone, _shape_param_0);  _shape_param_0 = None
        mm: "bf16[8192, 768]" = torch.ops.aten.mm.default(view, arg1_1);  view = arg1_1 = None
        view_1: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm, _shape_param_1);  mm = _shape_param_1 = None
        add: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_1, arg2_1);  view_1 = arg2_1 = None
        div: "bf16[1024, 8, 768]" = torch.ops.aten.div.Tensor(add, 8.0);  add = None
        view_2: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(div, _shape_param_2);  div = _shape_param_2 = None
        permute_1: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_2, [1, 0, 2, 3]);  view_2 = None
        permute_2: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_1, [0, 2, 1, 3]);  permute_1 = None
        view_3: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_2, _shape_param_3);  permute_2 = _shape_param_3 = None
        view_4: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_3, _shape_param_4);  view_3 = _shape_param_4 = None
        as_strided: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_4, _shape_param_5, _shape_param_6);  view_4 = _shape_param_5 = _shape_param_6 = None
        unsqueeze: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided, 4);  as_strided = None
        clone_1: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze, memory_format = torch.contiguous_format);  unsqueeze = None
        view_5: "bf16[288, 512, 64]" = torch.ops.aten.view.default(clone_1, _shape_param_7);  clone_1 = _shape_param_7 = None
        view_6: "bf16[8192, 768]" = torch.ops.aten.view.default(clone, _shape_param_8);  _shape_param_8 = None
        mm_1: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_6, arg3_1);  view_6 = arg3_1 = None
        view_7: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_1, _shape_param_9);  mm_1 = _shape_param_9 = None
        add_1: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_7, arg4_1);  view_7 = arg4_1 = None
        view_8: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_1, _shape_param_10);  add_1 = _shape_param_10 = None
        permute_4: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_8, [1, 0, 2, 3]);  view_8 = None
        permute_5: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_4, [0, 2, 1, 3]);  permute_4 = None
        view_9: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_5, _shape_param_11);  permute_5 = _shape_param_11 = None
        view_10: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_9, _shape_param_12);  view_9 = _shape_param_12 = None
        as_strided_1: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_10, _shape_param_13, _shape_param_14);  view_10 = _shape_param_13 = _shape_param_14 = None
        unsqueeze_1: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_1, 4);  as_strided_1 = None
        permute_6: "bf16[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_1, [0, 1, 4, 2, 3]);  unsqueeze_1 = None
        permute_7: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_6, [0, 1, 4, 3, 2]);  permute_6 = None
        clone_3: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_11: "bf16[288, 64, 512]" = torch.ops.aten.view.default(clone_3, _shape_param_15);  clone_3 = _shape_param_15 = None
        view_12: "bf16[8192, 768]" = torch.ops.aten.view.default(clone, _shape_param_16);  clone = _shape_param_16 = None
        mm_2: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_12, arg5_1);  view_12 = arg5_1 = None
        view_13: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_2, _shape_param_17);  mm_2 = _shape_param_17 = None
        add_2: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_13, arg6_1);  view_13 = arg6_1 = None
        view_14: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_2, _shape_param_18);  add_2 = _shape_param_18 = None
        permute_9: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_14, [1, 0, 2, 3]);  view_14 = None
        permute_10: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_9, [0, 2, 1, 3]);  permute_9 = None
        view_15: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_10, _shape_param_19);  permute_10 = _shape_param_19 = None
        constant_pad_nd: "bf16[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_15, _shape_param_20, -1.0);  view_15 = _shape_param_20 = None
        as_strided_2: "bf16[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd, _shape_param_21, _shape_param_22);  constant_pad_nd = _shape_param_21 = _shape_param_22 = None
        unsqueeze_2: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_2, 4);  as_strided_2 = None
        clone_5: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_2, memory_format = torch.contiguous_format);  unsqueeze_2 = None
        view_16: "bf16[384, 768, 64]" = torch.ops.aten.view.default(clone_5, _shape_param_23);  clone_5 = _shape_param_23 = None
        return (view_5, view_11, view_16)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
