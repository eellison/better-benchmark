"""Curated MM repro. Pattern hash: be68fde27b61."""
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
    def forward(self, arg0_1: "bf16[8192, 512]", arg1_1: "bf16[512, 2048]", arg2_1: "i64[64]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        mm: "bf16[8192, 2048]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        relu: "bf16[8, 1024, 2048]" = torch.ops.aten.relu.default(view);  view = None
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg2_1, 31);  arg2_1 = None
        inductor_random: "f32[8, 1024, 2048]" = torch.ops.prims.inductor_random.default(_shape_param_1, inductor_lookup_seed, 'rand');  _shape_param_1 = inductor_lookup_seed = None
        convert_element_type: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[8, 1024, 2048]" = torch.ops.aten.gt.Scalar(convert_element_type, 0.1);  convert_element_type = None
        mul: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(gt, relu)
        mul_1: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        convert_element_type_1: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        view_1: "bf16[8192, 2048]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_2);  convert_element_type_1 = _shape_param_2 = None
        le: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        return (gt, view_1, le)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
