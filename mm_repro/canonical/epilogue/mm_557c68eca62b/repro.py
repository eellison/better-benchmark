"""Curated MM repro. Pattern hash: 557c68eca62b."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
from mm_replay import make_inputs_from_config, benchmark_repro_safe as benchmark_repro
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[768]", arg1_1: "bf16[8192, 768]", arg2_1: "bf16[768, 768]", arg3_1: "i64[36]", arg4_1: "f32[8, 1024, 768]", arg5_1: "f32[768]", arg6_1: "f32[768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        mm: "bf16[8192, 768]" = torch.ops.aten.mm.default(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
        view: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        add: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view, convert_element_type);  view = convert_element_type = None
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg3_1, 10);  arg3_1 = None
        inductor_random: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default(_shape_param_1, inductor_lookup_seed, 'rand');  _shape_param_1 = inductor_lookup_seed = None
        convert_element_type_1: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_1, 0.1);  convert_element_type_1 = None
        mul: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt, add);  add = None
        mul_1: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        add_1: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 1024, 1]" = var_mean[0]
        getitem_1: "f32[8, 1024, 1]" = var_mean[1];  var_mean = None
        add_2: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        mul_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2, arg5_1);  arg5_1 = None
        add_3: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_3, arg6_1);  mul_3 = arg6_1 = None
        convert_element_type_2: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16)
        view_1: "bf16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_2);  convert_element_type_2 = _shape_param_2 = None
        div: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        return (gt, mul_2, add_3, view_1, div)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
