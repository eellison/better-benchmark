"""Curated MM repro. Pattern hash: dde312150717."""
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
    def forward(self, arg0_1: "bf16[2048, 8, 49, 32]", arg1_1: "bf16[2048, 8, 32, 49]", arg2_1: "i64[49, 49]", arg3_1: "bf16[169, 8]", arg4_1: "bf16[2048, 8, 49, 32]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12):
        # No stacktrace found for following nodes
        mul: "bf16[2048, 8, 49, 32]" = torch.ops.aten.mul.Tensor(arg0_1, 0.1767766952966369);  arg0_1 = None
        clone: "bf16[2048, 8, 49, 32]" = torch.ops.aten.clone.default(mul, memory_format = torch.contiguous_format);  mul = None
        view: "bf16[16384, 49, 32]" = torch.ops.aten.view.default(clone, _shape_param_0);  clone = _shape_param_0 = None
        constant_pad_nd: "bf16[16384, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view, _shape_param_1);  view = _shape_param_1 = None
        clone_1: "bf16[2048, 8, 32, 49]" = torch.ops.aten.clone.default(arg1_1, memory_format = torch.contiguous_format);  arg1_1 = None
        view_1: "bf16[16384, 32, 49]" = torch.ops.aten.view.default(clone_1, _shape_param_2);  clone_1 = _shape_param_2 = None
        constant_pad_nd_1: "bf16[16384, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_1, _shape_param_3);  view_1 = _shape_param_3 = None
        bmm: "bf16[16384, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd, constant_pad_nd_1);  constant_pad_nd = constant_pad_nd_1 = None
        slice_1: "bf16[16384, 49, 56]" = torch.ops.aten.slice.Tensor(bmm, 1, 0, -7);  bmm = None
        slice_2: "bf16[16384, 49, 49]" = torch.ops.aten.slice.Tensor(slice_1, 2, 0, -7);  slice_1 = None
        view_2: "bf16[2048, 8, 49, 49]" = torch.ops.aten.view.default(slice_2, _shape_param_4);  slice_2 = _shape_param_4 = None
        view_3: "i64[2401]" = torch.ops.aten.view.default(arg2_1, [-1]);  arg2_1 = None
        index: "bf16[2401, 8]" = torch.ops.aten.index.Tensor(arg3_1, [view_3]);  arg3_1 = view_3 = None
        view_4: "bf16[49, 49, 8]" = torch.ops.aten.view.default(index, _shape_param_5);  index = _shape_param_5 = None
        permute_1: "bf16[8, 49, 49]" = torch.ops.aten.permute.default(view_4, [2, 0, 1]);  view_4 = None
        clone_2: "bf16[8, 49, 49]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        unsqueeze: "bf16[1, 8, 49, 49]" = torch.ops.aten.view.default(clone_2, _shape_param_6);  clone_2 = _shape_param_6 = None
        add: "bf16[2048, 8, 49, 49]" = torch.ops.aten.add.Tensor(view_2, unsqueeze);  view_2 = unsqueeze = None
        convert_element_type: "f32[2048, 8, 49, 49]" = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
        amax: "f32[2048, 8, 49, 1]" = torch.ops.aten.amax.default(convert_element_type, [3], True)
        sub: "f32[2048, 8, 49, 49]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[2048, 8, 49, 49]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[2048, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp, [3], True)
        div: "f32[2048, 8, 49, 49]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_1: "bf16[2048, 8, 49, 49]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        view_5: "bf16[16384, 49, 49]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_7);  convert_element_type_1 = _shape_param_7 = None
        constant_pad_nd_2: "bf16[16384, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_5, _shape_param_8);  view_5 = _shape_param_8 = None
        clone_3: "bf16[2048, 8, 49, 32]" = torch.ops.aten.clone.default(arg4_1, memory_format = torch.contiguous_format);  arg4_1 = None
        view_6: "bf16[16384, 49, 32]" = torch.ops.aten.view.default(clone_3, _shape_param_9);  clone_3 = _shape_param_9 = None
        constant_pad_nd_3: "bf16[16384, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_6, _shape_param_10);  view_6 = _shape_param_10 = None
        bmm_1: "bf16[16384, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_2, constant_pad_nd_3);  constant_pad_nd_2 = constant_pad_nd_3 = None
        slice_3: "bf16[16384, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_1, 1, 0, -7);  bmm_1 = None
        view_7: "bf16[2048, 8, 49, 32]" = torch.ops.aten.view.default(slice_3, _shape_param_11);  slice_3 = _shape_param_11 = None
        permute_2: "bf16[2048, 49, 8, 32]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
        clone_4: "bf16[2048, 49, 8, 32]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_9: "bf16[100352, 256]" = torch.ops.aten.view.default(clone_4, _shape_param_12);  clone_4 = _shape_param_12 = None
        return view_9

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
