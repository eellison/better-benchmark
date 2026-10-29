"""Curated MM repro. Pattern hash: 46b97ee63143."""
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
    def forward(self, arg0_1: "f32[4, 2048, 768]", arg1_1: "b8[8192, 768]", arg2_1: "bf16[768, 3072]", arg3_1: "bf16[8192, 3072]", _shape_param_0):
        # No stacktrace found for following nodes
        view: "f32[8192, 768]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "bf16[8192, 768]" = torch.ops.prims.convert_element_type.default(view, torch.bfloat16);  view = None
        convert_element_type_1: "bf16[8192, 768]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        mul: "bf16[8192, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_1: "bf16[8192, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, mul);  convert_element_type = mul = None
        mm: "bf16[8192, 3072]" = torch.ops.aten.mm.default(mul_1, arg2_1);  mul_1 = arg2_1 = None
        le: "b8[8192, 3072]" = torch.ops.aten.le.Scalar(arg3_1, 0);  arg3_1 = None
        full: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[8192, 3072]" = torch.ops.aten.where.self(le, full, mm);  le = full = mm = None
        permute: "bf16[3072, 8192]" = torch.ops.aten.permute.default(where, [1, 0])
        return (where, permute)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
