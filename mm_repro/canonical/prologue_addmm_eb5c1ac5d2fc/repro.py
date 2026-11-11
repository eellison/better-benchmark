"""Curated MM repro. Pattern hash: eb5c1ac5d2fc."""
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
    def forward(self, arg0_1: "i64[1, 512]", arg1_1: "f32[30522, 768]", arg2_1: "i64[32, 512]", arg3_1: "f32[512, 768]", arg4_1: "i64[1, 512]", arg5_1: "f32[2, 768]", arg6_1: "f32[768]", arg7_1: "f32[768]", arg8_1: "bf16[384]", arg9_1: "bf16[768, 384]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        expand: "i64[32, 512]" = torch.ops.aten.expand.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        embedding: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg1_1, arg2_1, 0);  arg1_1 = arg2_1 = None
        embedding_1: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg3_1, arg4_1);  arg3_1 = arg4_1 = None
        embedding_2: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg5_1, expand);  arg5_1 = expand = None
        add: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        add_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean[0]
        getitem_1: "f32[32, 512, 1]" = var_mean[1];  var_mean = None
        add_2: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, arg6_1);  mul = arg6_1 = None
        add_3: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_1, arg7_1);  mul_1 = arg7_1 = None
        inductor_seeds: "i64[25]" = torch.ops.prims.inductor_seeds.default(25, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0);  inductor_seeds = None
        inductor_random: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_1, inductor_lookup_seed, 'rand');  _shape_param_1 = inductor_lookup_seed = None
        gt: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random, 0.1);  inductor_random = None
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt, add_3);  gt = add_3 = None
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None
        convert_element_type: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        view: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type, _shape_param_2);  convert_element_type = _shape_param_2 = None
        addmm: "bf16[16384, 384]" = torch.ops.aten.addmm.default(arg8_1, view, arg9_1);  arg8_1 = view = arg9_1 = None
        return addmm

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
