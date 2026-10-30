"""Curated MM repro. Pattern hash: c1320ad7d7f8."""
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
    def forward(self, arg0_1: "f32[32, 512, 1]", arg1_1: "f32[32, 512, 768]", arg2_1: "f32[32, 512, 1]", arg3_1: "f32[768]", arg4_1: "f32[768]", arg5_1: "f32[384]", arg6_1: "f32[768, 384]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10):
        # No stacktrace found for following nodes
        add: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(arg0_1, 1e-12);  arg0_1 = None
        rsqrt: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
        add_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
        inductor_seeds: "i64[25]" = torch.ops.prims.inductor_seeds.default(25, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0);  inductor_seeds = None
        inductor_random: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_0, inductor_lookup_seed, 'rand');  _shape_param_0 = inductor_lookup_seed = None
        gt: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random, 0.1);  inductor_random = None
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt, add_1);  gt = add_1 = None
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None
        convert_element_type: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        view: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type, _shape_param_1);  convert_element_type = _shape_param_1 = None
        convert_element_type_1: "bf16[384]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        convert_element_type_2: "bf16[768, 384]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.bfloat16);  arg6_1 = None
        addmm: "bf16[16384, 384]" = torch.ops.aten.addmm.default(convert_element_type_1, view, convert_element_type_2);  convert_element_type_1 = view = convert_element_type_2 = None
        view_1: "bf16[32, 512, 384]" = torch.ops.aten.view.default(addmm, _shape_param_2);  addmm = _shape_param_2 = None
        permute_1: "bf16[32, 384, 512]" = torch.ops.aten.permute.default(view_1, [0, 2, 1]);  view_1 = None
        clone: "bf16[32, 384, 512]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        unsqueeze: "bf16[32, 384, 512, 1]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        iota: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_1: "i64[1, 512]" = torch.ops.aten.view.default(iota, _shape_param_4);  iota = _shape_param_4 = None
        iota_1: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_2: "i64[9, 1]" = torch.ops.aten.view.default(iota_1, _shape_param_5);  iota_1 = _shape_param_5 = None
        add_2: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_1, unsqueeze_2);  unsqueeze_1 = unsqueeze_2 = None
        full: "i64[1, 1]" = torch.ops.aten.full.default([1, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        constant_pad_nd: "bf16[32, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze, _shape_param_6, 0.0);  unsqueeze = _shape_param_6 = None
        unsqueeze_4: "i64[9, 512, 1, 1]" = torch.ops.aten.view.default(add_2, _shape_param_7);  add_2 = _shape_param_7 = None
        index: "bf16[32, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(constant_pad_nd, [None, None, unsqueeze_4, full]);  constant_pad_nd = unsqueeze_4 = full = None
        permute_2: "bf16[32, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index, [0, 1, 2, 4, 3, 5]);  index = None
        view_2: "bf16[32, 3456, 512]" = torch.ops.aten.view.default(permute_2, _shape_param_8);  permute_2 = _shape_param_8 = None
        permute_3: "bf16[32, 512, 3456]" = torch.ops.aten.permute.default(view_2, [0, 2, 1]);  view_2 = None
        view_3: "bf16[32, 512, 384, 9]" = torch.ops.aten.view.default(permute_3, _shape_param_9);  permute_3 = _shape_param_9 = None
        clone_1: "bf16[32, 512, 384, 9]" = torch.ops.aten.clone.default(view_3, memory_format = torch.contiguous_format);  view_3 = None
        view_4: "bf16[98304, 64, 9]" = torch.ops.aten.view.default(clone_1, _shape_param_10);  clone_1 = _shape_param_10 = None
        permute_4: "bf16[98304, 9, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1])
        return (view_4, permute_4)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
