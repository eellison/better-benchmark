"""Curated MM repro. Pattern hash: 2d8fd7e87504."""
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
    def forward(self, arg0_1: "i64[1, 512]", arg1_1: "f32[30522, 128]", arg2_1: "i64[256, 128]", arg3_1: "f32[512]", arg4_1: "f32[384, 512]", arg5_1: "f32[512, 512]", arg6_1: "f32[2, 512]", arg7_1: "f32[512]", arg8_1: "f32[512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        slice_1: "i64[1, 128]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 0, 128);  arg0_1 = None
        full: "i64[256, 128]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        embedding: "f32[256, 128, 128]" = torch.ops.aten.embedding.default(arg1_1, arg2_1, 0);  arg1_1 = arg2_1 = None
        slice_2: "f32[256, 127, 128]" = torch.ops.aten.slice.Tensor(embedding, 1, 1, 9223372036854775807)
        constant_pad_nd: "f32[256, 128, 128]" = torch.ops.aten.constant_pad_nd.default(slice_2, [0, 0, 0, 1, 0, 0], 0.0);  slice_2 = None
        slice_3: "f32[256, 127, 128]" = torch.ops.aten.slice.Tensor(embedding, 1, 0, -1)
        constant_pad_nd_1: "f32[256, 128, 128]" = torch.ops.aten.constant_pad_nd.default(slice_3, [0, 0, 1, 0, 0, 0], 0.0);  slice_3 = None
        cat: "f32[256, 128, 384]" = torch.ops.aten.cat.default([constant_pad_nd, embedding, constant_pad_nd_1], 2);  constant_pad_nd = embedding = constant_pad_nd_1 = None
        convert_element_type: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        convert_element_type_1: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.bfloat16);  arg4_1 = None
        convert_element_type_2: "bf16[256, 128, 384]" = torch.ops.prims.convert_element_type.default(cat, torch.bfloat16);  cat = None
        view: "bf16[32768, 384]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_1);  convert_element_type_2 = _shape_param_1 = None
        addmm: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type, view, convert_element_type_1);  convert_element_type = view = convert_element_type_1 = None
        view_1: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm, _shape_param_2);  _shape_param_2 = None
        embedding_1: "f32[1, 128, 512]" = torch.ops.aten.embedding.default(arg5_1, slice_1);  arg5_1 = slice_1 = None
        embedding_2: "f32[256, 128, 512]" = torch.ops.aten.embedding.default(arg6_1, full);  arg6_1 = full = None
        add: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_1, embedding_1);  view_1 = embedding_1 = None
        add_1: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
        mul: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_1, arg7_1);  add_1 = arg7_1 = None
        add_2: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul, arg8_1);  mul = arg8_1 = None
        convert_element_type_3: "bf16[256, 128, 512]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16)
        view_2: "bf16[32768, 512]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_3);  convert_element_type_3 = _shape_param_3 = None
        return (addmm, add_2, view_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
