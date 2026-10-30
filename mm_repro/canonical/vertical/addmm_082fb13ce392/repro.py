"""Curated MM repro. Pattern hash: 082fb13ce392."""
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
    def forward(self, arg0_1: "i64[1, 512]", arg1_1: "i64[1, 512]", arg2_1: "f32[30522, 128]", arg3_1: "i64[64, 512]", arg4_1: "f32[2, 128]", arg5_1: "f32[512, 128]", arg6_1: "f32[128]", arg7_1: "f32[128]", arg8_1: "f32[256]", arg9_1: "f32[128, 256]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        gather: "i64[1, 512]" = torch.ops.aten.gather.default(arg0_1, 1, arg1_1);  arg0_1 = None
        expand_1: "i64[64, 512]" = torch.ops.aten.expand.default(gather, _shape_param_0);  gather = _shape_param_0 = None
        embedding: "f32[64, 512, 128]" = torch.ops.aten.embedding.default(arg2_1, arg3_1, 0);  arg2_1 = arg3_1 = None
        embedding_1: "f32[64, 512, 128]" = torch.ops.aten.embedding.default(arg4_1, expand_1);  arg4_1 = expand_1 = None
        add: "f32[64, 512, 128]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        embedding_2: "f32[1, 512, 128]" = torch.ops.aten.embedding.default(arg5_1, arg1_1);  arg5_1 = arg1_1 = None
        add_1: "f32[64, 512, 128]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem: "f32[64, 512, 1]" = var_mean[0]
        getitem_1: "f32[64, 512, 1]" = var_mean[1];  var_mean = None
        add_2: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub: "f32[64, 512, 128]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        mul: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul, arg6_1);  arg6_1 = None
        add_3: "f32[64, 512, 128]" = torch.ops.aten.add.Tensor(mul_1, arg7_1);  mul_1 = arg7_1 = None
        inductor_seeds: "i64[37]" = torch.ops.prims.inductor_seeds.default(37, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0);  inductor_seeds = None
        inductor_random: "f32[64, 512, 128]" = torch.ops.prims.inductor_random.default(_shape_param_1, inductor_lookup_seed, 'rand');  _shape_param_1 = inductor_lookup_seed = None
        gt: "b8[64, 512, 128]" = torch.ops.aten.gt.Scalar(inductor_random, 0.1);  inductor_random = None
        mul_2: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(gt, add_3);  gt = add_3 = None
        mul_3: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None
        convert_element_type: "bf16[256]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.bfloat16);  arg8_1 = None
        convert_element_type_1: "bf16[128, 256]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.bfloat16);  arg9_1 = None
        convert_element_type_2: "bf16[64, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        view: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_2);  convert_element_type_2 = _shape_param_2 = None
        addmm: "bf16[32768, 256]" = torch.ops.aten.addmm.default(convert_element_type, view, convert_element_type_1);  convert_element_type = convert_element_type_1 = None
        view_1: "bf16[64, 512, 256]" = torch.ops.aten.view.default(addmm, _shape_param_3);  addmm = _shape_param_3 = None
        div: "f32[64, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 128);  rsqrt = None
        return (mul, view, view_1, div)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
