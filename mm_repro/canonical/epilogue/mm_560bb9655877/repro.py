"""Curated MM repro. Pattern hash: 560bb9655877."""
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
    def forward(self, arg0_1: "f32[384, 512]", arg1_1: "bf16[4096, 384]", arg2_1: "i64[84]", arg3_1: "f32[32, 128, 512]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        mm: "bf16[4096, 512]" = torch.ops.aten.mm.default(arg1_1, convert_element_type);  arg1_1 = convert_element_type = None
        view: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg2_1, 0);  arg2_1 = None
        inductor_random: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_1, inductor_lookup_seed, 'rand');  _shape_param_1 = inductor_lookup_seed = None
        convert_element_type_1: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_1, 0.1);  convert_element_type_1 = None
        mul: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt, view);  gt = view = None
        mul_1: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        add: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(arg3_1, mul_1);  arg3_1 = mul_1 = None
        pow_1: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add, 2)
        return (add, pow_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
