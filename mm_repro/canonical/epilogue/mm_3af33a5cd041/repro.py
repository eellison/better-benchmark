"""Curated MM repro. Pattern hash: 3af33a5cd041."""
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
    def forward(self, arg0_1: "bf16[32000, 8192]", arg1_1: "bf16[8192, 1024]", arg2_1: "bf16[8192, 1024]", arg3_1: "f32[512, 16, 1024]", arg4_1: "bf16[8192, 1024]", arg5_1: "bf16[8192, 1024]", arg6_1: "b8[512, 16, 1024]", arg7_1: "i64[512, 16]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        mm: "bf16[32000, 1024]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[32000, 1024]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        unsqueeze: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(arg2_1, 0);  arg2_1 = None
        view: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.view.default(unsqueeze, _shape_param_0);  unsqueeze = _shape_param_0 = None
        squeeze: "bf16[512, 16, 1024, 1]" = torch.ops.aten.squeeze.dim(view, 4);  view = None
        squeeze_1: "bf16[512, 16, 1024]" = torch.ops.aten.squeeze.dim(squeeze, 3);  squeeze = None
        convert_element_type_1: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(squeeze_1, torch.float32);  squeeze_1 = None
        add: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(arg3_1, convert_element_type_1);  arg3_1 = convert_element_type_1 = None
        unsqueeze_1: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(arg4_1, 0);  arg4_1 = None
        view_1: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.view.default(unsqueeze_1, _shape_param_1);  unsqueeze_1 = _shape_param_1 = None
        squeeze_2: "bf16[512, 16, 1024, 1]" = torch.ops.aten.squeeze.dim(view_1, 4);  view_1 = None
        squeeze_3: "bf16[512, 16, 1024]" = torch.ops.aten.squeeze.dim(squeeze_2, 3);  squeeze_2 = None
        convert_element_type_2: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(squeeze_3, torch.float32);  squeeze_3 = None
        add_1: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(add, convert_element_type_2);  add = convert_element_type_2 = None
        unsqueeze_2: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(arg5_1, 0);  arg5_1 = None
        view_2: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.view.default(unsqueeze_2, _shape_param_2);  unsqueeze_2 = _shape_param_2 = None
        squeeze_4: "bf16[512, 16, 1024, 1]" = torch.ops.aten.squeeze.dim(view_2, 4);  view_2 = None
        squeeze_5: "bf16[512, 16, 1024]" = torch.ops.aten.squeeze.dim(squeeze_4, 3);  squeeze_4 = None
        convert_element_type_3: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(squeeze_5, torch.float32);  squeeze_5 = None
        add_2: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(add_1, convert_element_type_3);  add_1 = convert_element_type_3 = None
        convert_element_type_4: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.float32);  arg6_1 = None
        mul: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
        mul_1: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_2, mul);  add_2 = mul = None
        ge: "b8[512, 16]" = torch.ops.aten.ge.Scalar(arg7_1, 0)
        lt: "b8[512, 16]" = torch.ops.aten.lt.Scalar(arg7_1, 32000)
        bitwise_and: "b8[512, 16]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne: "b8[512, 16]" = torch.ops.aten.ne.Scalar(arg7_1, -1)
        bitwise_and_1: "b8[512, 16]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne);  bitwise_and = ne = None
        unsqueeze_3: "b8[512, 16, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full: "f32[32000, 1024]" = torch.ops.aten.full.default(_shape_param_3, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_3 = None
        _unsafe_masked_index_put_accumulate: "f32[32000, 1024]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full, unsqueeze_3, [arg7_1], mul_1);  full = unsqueeze_3 = arg7_1 = mul_1 = None
        add_3: "f32[32000, 1024]" = torch.ops.aten.add.Tensor(convert_element_type, _unsafe_masked_index_put_accumulate);  convert_element_type = _unsafe_masked_index_put_accumulate = None
        return add_3

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
