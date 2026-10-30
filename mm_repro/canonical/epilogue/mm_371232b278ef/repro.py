"""Curated MM repro. Pattern hash: 371232b278ef."""
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
    def forward(self, arg0_1: "bf16[8192, 1024]", arg1_1: "bf16[1024, 1024]", arg2_1: "f32[512, 16, 1024]", arg3_1: "f32[512, 16, 1024]", arg4_1: "b8[512, 16, 1024]", arg5_1: "i64[512, 16]", arg6_1: "f32[32000, 1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        mm: "bf16[8192, 1024]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        squeeze_1: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(squeeze_1, torch.float32);  squeeze_1 = None
        add: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(arg2_1, convert_element_type);  arg2_1 = convert_element_type = None
        add_1: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(add, arg3_1);  add = arg3_1 = None
        convert_element_type_1: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float32);  arg4_1 = None
        mul: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_1: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_1, mul);  add_1 = mul = None
        ge: "b8[512, 16]" = torch.ops.aten.ge.Scalar(arg5_1, 0)
        lt: "b8[512, 16]" = torch.ops.aten.lt.Scalar(arg5_1, 32000)
        bitwise_and: "b8[512, 16]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne: "b8[512, 16]" = torch.ops.aten.ne.Scalar(arg5_1, -1)
        bitwise_and_1: "b8[512, 16]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne);  bitwise_and = ne = None
        unsqueeze_1: "b8[512, 16, 1]" = torch.ops.aten.view.default(bitwise_and_1, _shape_param_1);  bitwise_and_1 = _shape_param_1 = None
        full: "f32[32000, 1024]" = torch.ops.aten.full.default(_shape_param_2, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_2 = None
        _unsafe_masked_index_put_accumulate: "f32[32000, 1024]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full, unsqueeze_1, [arg5_1], mul_1);  full = unsqueeze_1 = arg5_1 = mul_1 = None
        add_2: "f32[32000, 1024]" = torch.ops.aten.add.Tensor(arg6_1, _unsafe_masked_index_put_accumulate);  arg6_1 = _unsafe_masked_index_put_accumulate = None
        return add_2

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
