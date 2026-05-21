"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_train_000
Pattern hash: 550595226896
Shape hash: 40cd047e
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([8, 1024], i64), T([8, 1024], i32), T([50265, 768], f32), T([8, 1024], i64, gen=Index(50265)), T([4098, 768], f32), T([1, 768], f32), T([768], f32), T([768], f32))"

class Repro(torch.nn.Module):
    def forward(self, cumsum: "i64[8, 1024]", convert_element_type: "i32[8, 1024]", arg1_1: "f32[50265, 768]", arg0_1: "i64[8, 1024]", arg2_1: "f32[4098, 768]", arg3_1: "f32[1, 768]", arg4_1: "f32[768]", arg5_1: "f32[768]"):
        # No stacktrace found for following nodes
        full_default: "i64[8, 1024]" = torch.ops.aten.full.default([8, 1024], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_default: "i32[8, 1024]" = torch.ops.prims.convert_element_type.default(cumsum, torch.int32);  cumsum = None
        mul_tensor: "i32[8, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, convert_element_type);  convert_element_type_default = convert_element_type = None
        convert_element_type_default_1: "i64[8, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.int64);  mul_tensor = None
        add_tensor: "i64[8, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1);  convert_element_type_default_1 = None
        embedding_default: "f32[8, 1024, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 1);  arg1_1 = arg0_1 = None
        embedding_default_1: "f32[8, 1024, 768]" = torch.ops.aten.embedding.default(arg2_1, add_tensor, 1);  arg2_1 = add_tensor = None
        embedding_default_2: "f32[8, 1024, 768]" = torch.ops.aten.embedding.default(arg3_1, full_default);  arg3_1 = full_default = None
        add_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None
        add_tensor_2: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_tensor_1, embedding_default_2);  add_tensor_1 = embedding_default_2 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_2, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 1024, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 1024, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_3: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor_3);  add_tensor_3 = None
        sub_tensor: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_tensor_2, getitem_1);  add_tensor_2 = getitem_1 = None
        mul_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg4_1);  mul_tensor_1 = arg4_1 = None
        add_tensor_4: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_tensor_2, arg5_1);  mul_tensor_2 = arg5_1 = None
        inductor_seeds_default: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default, 1e-30);  inductor_random_default = None
        mul_tensor_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_scalar, add_tensor_4);  gt_scalar = add_tensor_4 = None
        mul_tensor_4: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 1.0);  mul_tensor_3 = None
        div_tensor: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_default, 768);  rsqrt_default = None
        return (mul_tensor_4, div_tensor)



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
