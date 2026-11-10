"""Curated MM repro. Pattern hash: ffd8eec946ce."""
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
    def forward(self, arg0_1: "bf16[768]", arg1_1: "bf16[2048, 3072]", arg2_1: "bf16[3072, 768]", arg3_1: "i64[61]", arg4_1: "f32[16, 128, 768]", arg5_1: "f32[768]", arg6_1: "f32[768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        addmm: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg0_1, arg1_1, arg2_1);  arg0_1 = arg1_1 = arg2_1 = None
        view: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg3_1, 9)
        inductor_random: "f32[16, 128, 768]" = torch.ops.prims.inductor_random.default(_shape_param_1, inductor_lookup_seed, 'rand');  _shape_param_1 = inductor_lookup_seed = None
        convert_element_type: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[16, 128, 768]" = torch.ops.aten.gt.Scalar(convert_element_type, 0.1);  convert_element_type = None
        mul: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(gt, view);  view = None
        mul_1: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        add: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(arg4_1, mul_1);  arg4_1 = mul_1 = None
        inductor_lookup_seed_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg3_1, 10);  arg3_1 = None
        inductor_random_1: "f32[16, 128, 768]" = torch.ops.prims.inductor_random.default(_shape_param_2, inductor_lookup_seed_1, 'rand');  _shape_param_2 = inductor_lookup_seed_1 = None
        gt_1: "b8[16, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_1, 0.1);  inductor_random_1 = None
        mul_2: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(gt_1, add);  add = None
        mul_3: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None
        mean: "f32[16, 128, 1]" = torch.ops.aten.mean.dim(mul_3, [-1], True)
        var: "f32[16, 128, 1]" = torch.ops.aten.var.correction(mul_3, [-1], correction = 1.0, keepdim = True)
        sqrt: "f32[16, 128, 1]" = torch.ops.aten.sqrt.default(var);  var = None
        sub: "f32[16, 128, 768]" = torch.ops.aten.sub.Tensor(mul_3, mean);  mean = None
        mul_4: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg5_1, sub);  arg5_1 = None
        add_1: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(sqrt, 1e-06)
        div: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_4, add_1);  mul_4 = add_1 = None
        add_2: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(div, arg6_1);  div = arg6_1 = None
        convert_element_type_1: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        view_1: "bf16[2048, 768]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_3);  convert_element_type_1 = _shape_param_3 = None
        return (gt, gt_1, mul_3, sqrt, sub, view_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
