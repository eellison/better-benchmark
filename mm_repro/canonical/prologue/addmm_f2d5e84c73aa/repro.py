"""Curated MM repro. Pattern hash: f2d5e84c73aa."""
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
    def forward(self, arg0_1: "f32[29056, 1024]", arg1_1: "i64[16, 512]", arg2_1: "f32[2, 1024]", arg3_1: "f32[512, 1024]", arg4_1: "i64[1, 512]", arg5_1: "f32[1024]", arg6_1: "f32[1024]", arg7_1: "bf16[1024]", arg8_1: "bf16[1024, 1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        full: "i64[16, 512]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        embedding: "f32[16, 512, 1024]" = torch.ops.aten.embedding.default(arg0_1, arg1_1, 0);  arg0_1 = arg1_1 = None
        embedding_1: "f32[16, 512, 1024]" = torch.ops.aten.embedding.default(arg2_1, full);  arg2_1 = full = None
        add: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        embedding_2: "f32[1, 512, 1024]" = torch.ops.aten.embedding.default(arg3_1, arg4_1);  arg3_1 = arg4_1 = None
        add_1: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
        inductor_seeds: "i64[49]" = torch.ops.prims.inductor_seeds.default(49, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0);  inductor_seeds = None
        inductor_random: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_1, inductor_lookup_seed, 'rand');  _shape_param_1 = inductor_lookup_seed = None
        gt: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random, 0.1);  inductor_random = None
        mul: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt, add_1);  gt = add_1 = None
        mul_1: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        var_mean = torch.ops.aten.var_mean.correction(mul_1, [2], correction = 0, keepdim = True)
        getitem: "f32[16, 512, 1]" = var_mean[0]
        getitem_1: "f32[16, 512, 1]" = var_mean[1];  var_mean = None
        add_2: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_1, getitem_1);  mul_1 = getitem_1 = None
        mul_2: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_3: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_2, arg5_1);  mul_2 = arg5_1 = None
        add_3: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_3, arg6_1);  mul_3 = arg6_1 = None
        convert_element_type: "bf16[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        view: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type, _shape_param_2);  convert_element_type = _shape_param_2 = None
        addmm: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg7_1, view, arg8_1);  arg7_1 = view = arg8_1 = None
        return addmm

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
