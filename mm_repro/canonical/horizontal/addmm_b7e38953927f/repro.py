"""Curated MM repro. Pattern hash: b7e38953927f."""
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
    def forward(self, arg0_1: "f32[32, 512, 1]", arg1_1: "f32[32, 512, 768]", arg2_1: "f32[32, 512, 1]", arg3_1: "f32[768]", arg4_1: "f32[768]", arg5_1: "f32[768, 768]", arg6_1: "bf16[768]", arg7_1: "f32[768, 768]", arg8_1: "bf16[768]", arg9_1: "f32[768, 768]", arg10_1: "bf16[768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        add: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(arg0_1, 1e-12);  arg0_1 = None
        rsqrt: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
        add_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
        inductor_seeds: "i64[37]" = torch.ops.prims.inductor_seeds.default(37, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0);  inductor_seeds = None
        inductor_random: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_0, inductor_lookup_seed, 'rand');  _shape_param_0 = inductor_lookup_seed = None
        gt: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random, 0.1);  inductor_random = None
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt, add_1);  gt = add_1 = None
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None
        convert_element_type: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        convert_element_type_1: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        view: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_1);  convert_element_type_1 = _shape_param_1 = None
        addmm: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg6_1, view, convert_element_type);  arg6_1 = convert_element_type = None
        view_2: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(addmm, _shape_param_2);  addmm = _shape_param_2 = None
        permute_1: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        convert_element_type_2: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.bfloat16);  arg7_1 = None
        addmm_1: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg8_1, view, convert_element_type_2);  arg8_1 = convert_element_type_2 = None
        view_4: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(addmm_1, _shape_param_3);  addmm_1 = _shape_param_3 = None
        permute_3: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        convert_element_type_3: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.bfloat16);  arg9_1 = None
        addmm_2: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg10_1, view, convert_element_type_3);  arg10_1 = view = convert_element_type_3 = None
        view_6: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(addmm_2, _shape_param_4);  addmm_2 = _shape_param_4 = None
        permute_5: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        mul_4: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_1, 0.3535533905932738);  permute_1 = None
        permute_6: "bf16[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        mul_5: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_6, 0.3535533905932738);  permute_6 = None
        clone: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(mul_4, memory_format = torch.contiguous_format);  mul_4 = None
        view_7: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone, _shape_param_5);  clone = _shape_param_5 = None
        clone_1: "bf16[32, 12, 64, 512]" = torch.ops.aten.clone.default(mul_5, memory_format = torch.contiguous_format);  mul_5 = None
        clone_2: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_8: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_2, _shape_param_6);  clone_2 = _shape_param_6 = None
        permute_7: "bf16[384, 64, 512]" = torch.ops.aten.permute.default(view_8, [0, 2, 1])
        return (view_7, clone_1, view_8, permute_7)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
