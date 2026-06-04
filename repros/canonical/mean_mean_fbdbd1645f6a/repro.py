"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_T5_base_train_000
Pattern hash: fbdbd1645f6a
Shape hash: 43298c8b
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([32128, 768], f32), T([8, 1024], i64, gen=Index(32128)), T([768], f32), T([8, 1024], i64), T([768], f32), S([8192, 768]), S([8192, 768]))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[32128, 768]", arg0_1: "i64[8, 1024]", arg2_1: "f32[768]", arg100_1: "i64[8, 1024]", arg101_1: "f32[768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        embedding_default: "f32[8, 1024, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg0_1 = None
        inductor_seeds_default: "i64[124]" = torch.ops.prims.inductor_seeds.default(124, device(type='cuda', index=0))
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_scalar, embedding_default);  gt_scalar = embedding_default = None
        mul_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        pow_tensor_scalar: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(mul_tensor_1, 2)
        mean_dim: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None
        add_tensor: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, rsqrt_default);  mul_tensor_1 = rsqrt_default = None
        mul_tensor_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg2_1, mul_tensor_2);  arg2_1 = mul_tensor_2 = None
        view_default: "f32[8192, 768]" = torch.ops.aten.view.default(mul_tensor_3, _shape_param_0);  mul_tensor_3 = _shape_param_0 = None
        full_default: "i64[8, 1024]" = torch.ops.aten.full.default([8, 1024], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor: "i64[8, 1023]" = torch.ops.aten.slice.Tensor(arg100_1, 1, 0, -1);  arg100_1 = None
        clone_default: "i64[8, 1023]" = torch.ops.aten.clone.default(slice_tensor);  slice_tensor = None
        slice_tensor_1: "i64[8, 1023]" = torch.ops.aten.slice.Tensor(full_default, 1, 1, 9223372036854775807)
        copy_default: "i64[8, 1023]" = torch.ops.aten.copy.default(slice_tensor_1, clone_default);  slice_tensor_1 = clone_default = None
        slice_scatter_default: "i64[8, 1024]" = torch.ops.aten.slice_scatter.default(full_default, copy_default, 1, 1, 9223372036854775807);  full_default = copy_default = None
        full_default_1: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        select_int: "i64[8]" = torch.ops.aten.select.int(slice_scatter_default, 1, 0)
        copy_default_1: "i64[8]" = torch.ops.aten.copy.default(select_int, full_default_1);  select_int = full_default_1 = None
        select_scatter_default: "i64[8, 1024]" = torch.ops.aten.select_scatter.default(slice_scatter_default, copy_default_1, 1, 0);  slice_scatter_default = copy_default_1 = None
        eq_scalar: "b8[8, 1024]" = torch.ops.aten.eq.Scalar(select_scatter_default, -100)
        full_default_2: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[8, 1024]" = torch.ops.aten.where.self(eq_scalar, full_default_2, select_scatter_default);  eq_scalar = full_default_2 = select_scatter_default = None
        embedding_default_1: "f32[8, 1024, 768]" = torch.ops.aten.embedding.default(arg1_1, where_self);  arg1_1 = where_self = None
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 50);  inductor_seeds_default = None
        inductor_random_default_1: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_scalar_1: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.1);  inductor_random_default_1 = None
        mul_tensor_4: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_scalar_1, embedding_default_1);  gt_scalar_1 = embedding_default_1 = None
        mul_tensor_5: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_4, 1.1111111111111112);  mul_tensor_4 = None
        pow_tensor_scalar_1: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(mul_tensor_5, 2)
        mean_dim_1: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar_1, [-1], True);  pow_tensor_scalar_1 = None
        add_tensor_1: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_dim_1, 1e-06);  mean_dim_1 = None
        rsqrt_default_1: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_6: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_5, rsqrt_default_1);  mul_tensor_5 = rsqrt_default_1 = None
        mul_tensor_7: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg101_1, mul_tensor_6);  arg101_1 = mul_tensor_6 = None
        view_default_1: "f32[8192, 768]" = torch.ops.aten.view.default(mul_tensor_7, _shape_param_1);  mul_tensor_7 = _shape_param_1 = None
        return (view_default, view_default_1)

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
