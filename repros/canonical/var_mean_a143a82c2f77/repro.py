"""
Standalone repro captured via capture_hook.
Label: hf_DistilBertForMaskedLM_train
Pattern hash: a143a82c2f77
Shape hash: 8e9bc156
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
    def forward(self, arg0_1: "f32[30522, 768]", arg1_1: "i64[256, 128]", arg2_1: "i64[1, 512]", arg3_1: "f32[512, 768]", arg4_1: "f32[768]", arg5_1: "f32[768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        embedding: "f32[256, 128, 768]" = torch.ops.aten.embedding.default(arg0_1, arg1_1, 0);  arg0_1 = arg1_1 = None
        slice_1: "i64[1, 128]" = torch.ops.aten.slice.Tensor(arg2_1, 1, 0, 128);  arg2_1 = None
        embedding_1: "f32[1, 128, 768]" = torch.ops.aten.embedding.default(arg3_1, slice_1);  arg3_1 = slice_1 = None
        add: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1)
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[256, 128, 1]" = var_mean[0]
        getitem_1: "f32[256, 128, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(add, getitem_1);  add = None
        mul: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
        add_2: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
        inductor_seeds: "i64[13]" = torch.ops.prims.inductor_seeds.default(13, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0)
        inductor_random: "f32[256, 128, 768]" = torch.ops.prims.inductor_random.default(_shape_param_0, inductor_lookup_seed, 'rand');  _shape_param_0 = inductor_lookup_seed = None
        gt: "b8[256, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random, 0.1);  inductor_random = None
        mul_2: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(gt, add_2);  add_2 = None
        mul_3: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None
        convert_element_type: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16)
        view: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type, _shape_param_1);  convert_element_type = _shape_param_1 = None
        return (embedding, embedding_1, getitem_1, rsqrt, inductor_seeds, gt, mul_3, view)



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
