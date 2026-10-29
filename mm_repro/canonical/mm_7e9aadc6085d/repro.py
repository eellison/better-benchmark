"""Curated MM repro. Pattern hash: 7e9aadc6085d."""
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
    def forward(self, arg0_1: "bf16[128, 1000]", arg1_1: "bf16[1000, 2304]", arg2_1: "bf16[128, 2304, 7, 7]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        mm: "bf16[128, 2304]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        full: "bf16[294912]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        as_strided_scatter: "bf16[294912]" = torch.ops.aten.as_strided_scatter.default(full, mm, _shape_param_1, _shape_param_2, 0);  full = mm = _shape_param_1 = _shape_param_2 = None
        as_strided: "bf16[128, 2304, 1, 1]" = torch.ops.aten.as_strided.default(as_strided_scatter, _shape_param_3, _shape_param_4, 0);  as_strided_scatter = _shape_param_3 = _shape_param_4 = None
        expand: "bf16[128, 2304, 7, 7]" = torch.ops.aten.expand.default(as_strided, _shape_param_5);  as_strided = _shape_param_5 = None
        div: "bf16[128, 2304, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
        convert_element_type: "f32[128, 2304, 7, 7]" = torch.ops.prims.convert_element_type.default(div, torch.float32);  div = None
        convert_element_type_1: "f32[128, 2304, 7, 7]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        sigmoid: "f32[128, 2304, 7, 7]" = torch.ops.aten.sigmoid.default(convert_element_type_1)
        mul: "f32[128, 2304, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type, sigmoid);  convert_element_type = None
        sub: "f32[128, 2304, 7, 7]" = torch.ops.aten.sub.Tensor(1, sigmoid);  sigmoid = None
        mul_1: "f32[128, 2304, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type_1, sub);  convert_element_type_1 = sub = None
        add: "f32[128, 2304, 7, 7]" = torch.ops.aten.add.Tensor(mul_1, 1);  mul_1 = None
        mul_2: "f32[128, 2304, 7, 7]" = torch.ops.aten.mul.Tensor(mul, add);  mul = add = None
        convert_element_type_2: "bf16[128, 2304, 7, 7]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        return convert_element_type_2

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
