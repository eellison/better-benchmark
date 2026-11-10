"""Curated MM repro. Pattern hash: 509360d67f93."""
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
    def forward(self, arg0_1: "bf16[4096]", arg1_1: "bf16[64, 25088]", arg2_1: "bf16[25088, 4096]", _shape_param_0):
        # No stacktrace found for following nodes
        addmm: "bf16[64, 4096]" = torch.ops.aten.addmm.default(arg0_1, arg1_1, arg2_1);  arg0_1 = arg1_1 = arg2_1 = None
        relu: "bf16[64, 4096]" = torch.ops.aten.relu.default(addmm);  addmm = None
        inductor_seeds: "i64[2]" = torch.ops.prims.inductor_seeds.default(2, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0);  inductor_seeds = None
        inductor_random: "f32[64, 4096]" = torch.ops.prims.inductor_random.default(_shape_param_0, inductor_lookup_seed, 'rand');  _shape_param_0 = inductor_lookup_seed = None
        convert_element_type: "bf16[64, 4096]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[64, 4096]" = torch.ops.aten.gt.Scalar(convert_element_type, 0.5);  convert_element_type = None
        mul: "bf16[64, 4096]" = torch.ops.aten.mul.Tensor(gt, relu);  gt = None
        mul_1: "bf16[64, 4096]" = torch.ops.aten.mul.Tensor(mul, 2.0);  mul = None
        le: "b8[64, 4096]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        return (mul_1, le)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
