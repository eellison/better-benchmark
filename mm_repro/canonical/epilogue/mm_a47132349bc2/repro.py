"""Curated MM repro. Pattern hash: a47132349bc2."""
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
    def forward(self, arg0_1: "bf16[32768, 512]", arg1_1: "bf16[512, 384]", arg2_1: "i64[256, 128]", arg3_1: "f32[30522, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        mm: "bf16[32768, 384]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[256, 128, 384]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[256, 128, 384]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        slice_1: "f32[256, 128, 128]" = torch.ops.aten.slice.Tensor(convert_element_type, 2, 0, 128)
        slice_2: "f32[256, 128, 128]" = torch.ops.aten.slice.Tensor(convert_element_type, 2, 128, 256)
        slice_3: "f32[256, 128, 128]" = torch.ops.aten.slice.Tensor(convert_element_type, 2, 256, 384);  convert_element_type = None
        constant_pad_nd: "f32[256, 127, 128]" = torch.ops.aten.constant_pad_nd.default(slice_3, [0, 0, -1, 0, 0, 0]);  slice_3 = None
        full: "f32[256, 128, 128]" = torch.ops.aten.full.default(_shape_param_1, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        slice_scatter: "f32[256, 128, 128]" = torch.ops.aten.slice_scatter.default(full, constant_pad_nd, 1, 0, -1);  constant_pad_nd = None
        add: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(slice_2, slice_scatter);  slice_2 = slice_scatter = None
        constant_pad_nd_1: "f32[256, 127, 128]" = torch.ops.aten.constant_pad_nd.default(slice_1, [0, 0, 0, -1, 0, 0]);  slice_1 = None
        slice_scatter_1: "f32[256, 128, 128]" = torch.ops.aten.slice_scatter.default(full, constant_pad_nd_1, 1, 1, 9223372036854775807);  full = constant_pad_nd_1 = None
        add_1: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(add, slice_scatter_1);  add = slice_scatter_1 = None
        ge: "b8[256, 128]" = torch.ops.aten.ge.Scalar(arg2_1, 0)
        lt: "b8[256, 128]" = torch.ops.aten.lt.Scalar(arg2_1, 30522)
        bitwise_and: "b8[256, 128]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne: "b8[256, 128]" = torch.ops.aten.ne.Scalar(arg2_1, 0)
        bitwise_and_1: "b8[256, 128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne);  bitwise_and = ne = None
        unsqueeze: "b8[256, 128, 1]" = torch.ops.aten.view.default(bitwise_and_1, _shape_param_2);  bitwise_and_1 = _shape_param_2 = None
        full_1: "f32[30522, 128]" = torch.ops.aten.full.default(_shape_param_3, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_3 = None
        _unsafe_masked_index_put_accumulate: "f32[30522, 128]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_1, unsqueeze, [arg2_1], add_1);  full_1 = unsqueeze = arg2_1 = add_1 = None
        add_2: "f32[30522, 128]" = torch.ops.aten.add.Tensor(arg3_1, _unsafe_masked_index_put_accumulate);  arg3_1 = _unsafe_masked_index_put_accumulate = None
        return add_2

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
