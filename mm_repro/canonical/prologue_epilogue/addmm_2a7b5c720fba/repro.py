"""Curated MM repro. Pattern hash: 2a7b5c720fba."""
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
    def forward(self, arg0_1: "f32[8, 512, 1]", arg1_1: "f32[8, 512, 1536]", arg2_1: "f32[8, 512, 1]", arg3_1: "f32[1536]", arg4_1: "f32[1536]", arg5_1: "f32[1536]", arg6_1: "f32[1536, 1536]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        add: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(arg0_1, 1e-07);  arg0_1 = None
        rsqrt: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
        mul: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
        add_1: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
        inductor_seeds: "i64[73]" = torch.ops.prims.inductor_seeds.default(73, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0);  inductor_seeds = None
        inductor_random: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_0, inductor_lookup_seed, 'rand');  _shape_param_0 = inductor_lookup_seed = None
        gt: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random, 0.1);  inductor_random = None
        mul_2: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt, add_1);  gt = add_1 = None
        mul_3: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None
        convert_element_type: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        view: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type, _shape_param_1);  convert_element_type = _shape_param_1 = None
        convert_element_type_1: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        convert_element_type_2: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.bfloat16);  arg6_1 = None
        addmm: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_1, view, convert_element_type_2);  convert_element_type_1 = view = convert_element_type_2 = None
        view_2: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(addmm, _shape_param_2);  addmm = _shape_param_2 = None
        permute_1: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        clone: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_3: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        permute_2: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_3, [0, 2, 1]);  view_3 = None
        full: "bf16[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_2, full);  permute_2 = full = None
        permute_3: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div, [0, 2, 1])
        return (div, permute_3)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
