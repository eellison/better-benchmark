"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_train
Pattern hash: ac2b7ba56267
Shape hash: 496feaec
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
    def forward(self, arg0_1: "i64[8, 1024]", arg1_1: "i32[8, 1024]", arg2_1: "f32[50265, 768]", arg3_1: "i64[8, 1024]", arg4_1: "f32[4098, 768]", arg5_1: "f32[1, 768]", arg6_1: "f32[768]", arg7_1: "f32[768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        full: "i64[8, 1024]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        convert_element_type: "i32[8, 1024]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.int32);  arg0_1 = None
        mul: "i32[8, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, arg1_1);  convert_element_type = arg1_1 = None
        convert_element_type_1: "i64[8, 1024]" = torch.ops.prims.convert_element_type.default(mul, torch.int64);  mul = None
        add: "i64[8, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1);  convert_element_type_1 = None
        embedding: "f32[8, 1024, 768]" = torch.ops.aten.embedding.default(arg2_1, arg3_1, 1);  arg2_1 = arg3_1 = None
        embedding_1: "f32[8, 1024, 768]" = torch.ops.aten.embedding.default(arg4_1, add, 1);  arg4_1 = None
        embedding_2: "f32[8, 1024, 768]" = torch.ops.aten.embedding.default(arg5_1, full);  arg5_1 = None
        add_1: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        add_2: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_1, embedding_2);  add_1 = embedding_2 = None
        var_mean = torch.ops.aten.var_mean.correction(add_2, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 1024, 1]" = var_mean[0]
        getitem_1: "f32[8, 1024, 1]" = var_mean[1];  var_mean = None
        add_3: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        sub: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_2, getitem_1);  add_2 = getitem_1 = None
        mul_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1, arg6_1);  arg6_1 = None
        add_4: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_2, arg7_1);  mul_2 = arg7_1 = None
        inductor_seeds: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0);  inductor_seeds = None
        inductor_random: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default(_shape_param_1, inductor_lookup_seed, 'rand');  _shape_param_1 = inductor_lookup_seed = None
        gt: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random, 0.1);  inductor_random = None
        mul_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt, add_4);  add_4 = None
        mul_4: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_3, 1.1111111111111112);  mul_3 = None
        div: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        return (full, add, mul_1, gt, mul_4, div)



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
