"""Curated MM repro. Pattern hash: 39de45b1b1a3."""
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
    def forward(self, arg0_1: "bf16[64, 512, 14, 14]", arg1_1: "f32[4096]", arg2_1: "f32[25088, 4096]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        relu: "bf16[64, 512, 14, 14]" = torch.ops.aten.relu.default(arg0_1);  arg0_1 = None
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu, _shape_param_0, _shape_param_1, [0, 0], [1, 1], False);  relu = _shape_param_0 = _shape_param_1 = None
        getitem: "bf16[64, 512, 7, 7]" = _low_memory_max_pool_with_offsets[0];  _low_memory_max_pool_with_offsets = None
        view: "bf16[64, 25088]" = torch.ops.aten.view.default(getitem, _shape_param_2);  getitem = _shape_param_2 = None
        convert_element_type: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        convert_element_type_1: "bf16[25088, 4096]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        addmm: "bf16[64, 4096]" = torch.ops.aten.addmm.default(convert_element_type, view, convert_element_type_1);  convert_element_type = view = convert_element_type_1 = None
        relu_1: "bf16[64, 4096]" = torch.ops.aten.relu.default(addmm);  addmm = None
        inductor_seeds: "i64[2]" = torch.ops.prims.inductor_seeds.default(2, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0);  inductor_seeds = None
        inductor_random: "f32[64, 4096]" = torch.ops.prims.inductor_random.default(_shape_param_3, inductor_lookup_seed, 'rand');  _shape_param_3 = inductor_lookup_seed = None
        convert_element_type_2: "bf16[64, 4096]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[64, 4096]" = torch.ops.aten.gt.Scalar(convert_element_type_2, 0.5);  convert_element_type_2 = None
        mul: "bf16[64, 4096]" = torch.ops.aten.mul.Tensor(gt, relu_1);  gt = None
        mul_1: "bf16[64, 4096]" = torch.ops.aten.mul.Tensor(mul, 2.0);  mul = None
        le: "b8[64, 4096]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        return (mul_1, le)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
