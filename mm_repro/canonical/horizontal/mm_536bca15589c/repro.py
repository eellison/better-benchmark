"""Curated MM repro. Pattern hash: 536bca15589c."""
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
    def forward(self, arg0_1: "i64[16, 512]", arg1_1: "f32[32000, 1024]", arg2_1: "f32[1024, 16, 64]", arg3_1: "f32[1024, 16, 64]", arg4_1: "f32[1024, 16, 64]", arg5_1: "f32[16, 64]", arg6_1: "f32[16, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12):
        # No stacktrace found for following nodes
        permute: "i64[512, 16]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        clone: "i64[512, 16]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        embedding: "f32[512, 16, 1024]" = torch.ops.aten.embedding.default(arg1_1, clone);  arg1_1 = clone = None
        inductor_seeds: "i64[99]" = torch.ops.prims.inductor_seeds.default(99, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0);  inductor_seeds = None
        inductor_random: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_0, inductor_lookup_seed, 'rand');  _shape_param_0 = inductor_lookup_seed = None
        gt: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(inductor_random, 0.1);  inductor_random = None
        mul: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt, embedding);  gt = embedding = None
        mul_1: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        convert_element_type: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        convert_element_type_1: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        squeeze: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type, _shape_param_1);  convert_element_type = _shape_param_1 = None
        squeeze_1: "bf16[1024, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_2);  convert_element_type_1 = _shape_param_2 = None
        mm: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze, squeeze_1);  squeeze_1 = None
        view_2: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(mm, _shape_param_3);  mm = _shape_param_3 = None
        permute_1: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_2, [0, 1, 3, 4, 2]);  view_2 = None
        view_3: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_1, _shape_param_4);  permute_1 = _shape_param_4 = None
        convert_element_type_2: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        squeeze_2: "bf16[1024, 1024]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_5);  convert_element_type_2 = _shape_param_5 = None
        mm_1: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze, squeeze_2);  squeeze_2 = None
        view_5: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(mm_1, _shape_param_6);  mm_1 = _shape_param_6 = None
        permute_2: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_5, [0, 1, 3, 4, 2]);  view_5 = None
        view_6: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_2, _shape_param_7);  permute_2 = _shape_param_7 = None
        convert_element_type_3: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.bfloat16);  arg4_1 = None
        squeeze_3: "bf16[1024, 1024]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_8);  convert_element_type_3 = _shape_param_8 = None
        mm_2: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze, squeeze_3);  squeeze = squeeze_3 = None
        view_8: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(mm_2, _shape_param_9);  mm_2 = _shape_param_9 = None
        permute_3: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_8, [0, 1, 3, 4, 2]);  view_8 = None
        view_9: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_3, _shape_param_10);  permute_3 = _shape_param_10 = None
        add: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_3, arg5_1);  arg5_1 = None
        unsqueeze_11: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.view.default(view_6, _shape_param_11);  view_6 = _shape_param_11 = None
        add_1: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_3, arg6_1);  view_3 = arg6_1 = None
        unsqueeze_12: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.view.default(view_9, _shape_param_12);  view_9 = _shape_param_12 = None
        return (add, unsqueeze_11, add_1, unsqueeze_12)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
