"""
Standalone repro captured via capture_hook.
Label: hf_T5ForConditionalGeneration_train
Pattern hash: 996428153329
Shape hash: d37791c4
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
    def forward(self, arg0_1: "f32[32128, 512]", arg1_1: "i64[8, 1024]", arg2_1: "f32[512]", arg3_1: "f32[512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        embedding: "f32[8, 1024, 512]" = torch.ops.aten.embedding.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        inductor_seeds: "i64[64]" = torch.ops.prims.inductor_seeds.default(64, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0)
        inductor_random: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_0, inductor_lookup_seed, 'rand');  _shape_param_0 = inductor_lookup_seed = None
        gt: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random, 0.1);  inductor_random = None
        mul: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt, embedding)
        mul_1: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        pow_1: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(mul_1, 2)
        mean: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul_2: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_1, rsqrt)
        mul_3: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg2_1, mul_2);  arg2_1 = mul_2 = None
        convert_element_type: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        view: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type, _shape_param_1);  convert_element_type = _shape_param_1 = None
        inductor_lookup_seed_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 26)
        inductor_random_1: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_2, inductor_lookup_seed_1, 'rand');  _shape_param_2 = inductor_lookup_seed_1 = None
        gt_1: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_1, 0.1);  inductor_random_1 = None
        mul_4: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_1, embedding)
        mul_5: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_4, 1.1111111111111112);  mul_4 = None
        pow_2: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(mul_5, 2)
        mean_1: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
        add_1: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        rsqrt_1: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul_6: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_5, rsqrt_1)
        mul_7: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg3_1, mul_6);  arg3_1 = mul_6 = None
        convert_element_type_1: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16);  mul_7 = None
        view_1: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_3);  convert_element_type_1 = _shape_param_3 = None
        return (embedding, inductor_seeds, gt, mul_1, rsqrt, view, gt_1, mul_5, rsqrt_1, view_1)



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
