"""
Standalone repro captured via capture_hook.
Label: hf_XLNetLMHeadModel_train
Pattern hash: 4254ac4c0d96
Shape hash: 4ae3e770
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[16, 512]", arg1_1: "f32[32000, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        permute: "i64[512, 16]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        clone: "i64[512, 16]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        embedding: "f32[512, 16, 1024]" = torch.ops.aten.embedding.default(arg1_1, clone);  arg1_1 = None
        inductor_seeds: "i64[99]" = torch.ops.prims.inductor_seeds.default(99, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0)
        inductor_random: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_0, inductor_lookup_seed, 'rand');  _shape_param_0 = inductor_lookup_seed = None
        gt: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(inductor_random, 0.1);  inductor_random = None
        mul: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt, embedding);  embedding = None
        mul_1: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        iota: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type: "f32[512]" = torch.ops.prims.convert_element_type.default(iota, torch.float32);  iota = None
        div: "f32[512]" = torch.ops.aten.div.Tensor(convert_element_type, 1024);  convert_element_type = None
        pow_1: "f32[512]" = torch.ops.aten.pow.Scalar(10000, div);  div = None
        reciprocal: "f32[512]" = torch.ops.aten.reciprocal.default(pow_1);  pow_1 = None
        mul_2: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        iota_1: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 512, step = -1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_1: "f32[1024]" = torch.ops.prims.convert_element_type.default(iota_1, torch.float32);  iota_1 = None
        convert_element_type_2: "bf16[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.bfloat16);  convert_element_type_1 = None
        convert_element_type_3: "bf16[512]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        unsqueeze: "bf16[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1);  convert_element_type_2 = None
        permute_1: "bf16[1024, 1]" = torch.ops.aten.permute.default(unsqueeze, [0, 1]);  unsqueeze = None
        unsqueeze_1: "bf16[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1);  convert_element_type_3 = None
        permute_2: "bf16[1, 512]" = torch.ops.aten.permute.default(unsqueeze_1, [1, 0]);  unsqueeze_1 = None
        mul_3: "bf16[1024, 512]" = torch.ops.aten.mul.Tensor(permute_1, permute_2);  permute_1 = permute_2 = None
        sin: "bf16[1024, 512]" = torch.ops.aten.sin.default(mul_3)
        cos: "bf16[1024, 512]" = torch.ops.aten.cos.default(mul_3);  mul_3 = None
        cat: "bf16[1024, 1024]" = torch.ops.aten.cat.default([sin, cos], -1);  sin = cos = None
        unsqueeze_2: "bf16[1024, 1, 1024]" = torch.ops.aten.unsqueeze.default(cat, 1);  cat = None
        expand: "bf16[1024, 16, 1024]" = torch.ops.aten.expand.default(unsqueeze_2, _shape_param_1);  unsqueeze_2 = _shape_param_1 = None
        inductor_lookup_seed_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 1)
        inductor_random_1: "f32[1024, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_2, inductor_lookup_seed_1, 'rand');  _shape_param_2 = inductor_lookup_seed_1 = None
        convert_element_type_4: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_1, torch.bfloat16);  inductor_random_1 = None
        gt_1: "b8[1024, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_4, 0.1);  convert_element_type_4 = None
        mul_4: "bf16[1024, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_1, expand);  gt_1 = expand = None
        mul_5: "bf16[1024, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_4, 1.1111111111111112);  mul_4 = None
        convert_element_type_5: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16)
        unsqueeze_3: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_5, 3);  convert_element_type_5 = None
        unsqueeze_4: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 4);  unsqueeze_3 = None
        view: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_4, _shape_param_3);  unsqueeze_4 = _shape_param_3 = None
        squeeze: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view, 0)
        convert_element_type_6: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        unsqueeze_5: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_6, 3);  convert_element_type_6 = None
        unsqueeze_6: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 4);  unsqueeze_5 = None
        view_1: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_6, _shape_param_4);  unsqueeze_6 = _shape_param_4 = None
        squeeze_1: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_1, 0)
        permute_3: "bf16[1, 1024, 16384]" = torch.ops.aten.permute.default(view_1, [0, 2, 1]);  view_1 = None
        squeeze_2: "bf16[1024, 16384]" = torch.ops.aten.squeeze.dim(permute_3, 0);  permute_3 = None
        permute_4: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
        squeeze_3: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_4, 0);  permute_4 = None
        return (clone, inductor_seeds, gt, mul_1, squeeze, squeeze_1, squeeze_2, squeeze_3)



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
