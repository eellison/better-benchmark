"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Reformer_train_001
Pattern hash: 03cf69441b22
Shape hash: 91e2246e
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([320, 256], f32), T([8, 4096], i64, gen=Index(320)), T([64, 1, 64], f32), T([1, 64, 192], f32), S([8, 64, 64, 64]), S([8, 64, 64, 192]), S([8, 4096, -1]))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[320, 256]", arg0_1: "i64[8, 4096]", arg2_1: "f32[64, 1, 64]", arg3_1: "f32[1, 64, 192]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        embedding_default: "f32[8, 4096, 256]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        inductor_seeds_default: "i64[2]" = torch.ops.prims.inductor_seeds.default(2, device(type='cuda', index=0))
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default: "f32[8, 4096, 256]" = torch.ops.prims.inductor_random.default([8, 4096, 256], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[8, 4096, 256]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.05);  inductor_random_default = None
        mul_tensor: "f32[8, 4096, 256]" = torch.ops.aten.mul.Tensor(gt_scalar, embedding_default);  gt_scalar = embedding_default = None
        mul_tensor_1: "f32[8, 4096, 256]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.0526315789473684);  mul_tensor = None
        expand_default: "f32[8, 64, 64, 64]" = torch.ops.aten.expand.default(arg2_1, _shape_param_0);  arg2_1 = _shape_param_0 = None
        expand_default_1: "f32[8, 64, 64, 192]" = torch.ops.aten.expand.default(arg3_1, _shape_param_1);  arg3_1 = _shape_param_1 = None
        cat_default: "f32[8, 64, 64, 256]" = torch.ops.aten.cat.default([expand_default, expand_default_1], -1);  expand_default = expand_default_1 = None
        permute_default: "f32[8, 64, 64, 256]" = torch.ops.aten.permute.default(cat_default, [0, 2, 1, 3]);  cat_default = None
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1);  inductor_seeds_default = None
        inductor_random_default_1: "f32[8, 64, 1, 1]" = torch.ops.prims.inductor_random.default([8, 64, 1, 1], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        lt_scalar: "b8[8, 64, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_1, 0.95);  inductor_random_default_1 = None
        convert_element_type_default: "f32[8, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_scalar, torch.float32);  lt_scalar = None
        div_scalar: "f32[8, 64, 1, 1]" = torch.ops.aten.div.Scalar(convert_element_type_default, 0.95);  convert_element_type_default = None
        mul_tensor_2: "f32[8, 64, 64, 256]" = torch.ops.aten.mul.Tensor(permute_default, div_scalar);  permute_default = div_scalar = None
        permute_default_1: "f32[8, 64, 64, 256]" = torch.ops.aten.permute.default(mul_tensor_2, [0, 2, 1, 3]);  mul_tensor_2 = None
        view_default: "f32[8, 4096, 256]" = torch.ops.aten.view.default(permute_default_1, _shape_param_2);  permute_default_1 = _shape_param_2 = None
        add_tensor: "f32[8, 4096, 256]" = torch.ops.aten.add.Tensor(mul_tensor_1, view_default);  mul_tensor_1 = view_default = None
        return add_tensor

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
