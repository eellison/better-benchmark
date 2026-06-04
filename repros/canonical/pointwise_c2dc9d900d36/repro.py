"""
Standalone repro captured via capture_hook.
Label: vllm_facebook_opt-125m_002
Pattern hash: c2dc9d900d36
Shape hash: eedfbdb8
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([2], i64), T([2048, 768], f16), T([2048, 768], f16), S([4, 512, 768]))"

class Repro(torch.nn.Module):
    def forward(self, inductor_seeds: "i64[2]", addmm_5: "f16[2048, 768]", view_12: "f16[2048, 768]", _shape_param_0):
        # No stacktrace found for following nodes
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 1);  inductor_seeds = None
        inductor_random_default: "f32[2048, 768]" = torch.ops.prims.inductor_random.default([2048, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        convert_element_type_default: "f16[2048, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.float16);  inductor_random_default = None
        gt_scalar: "b8[2048, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_default, 0.1);  convert_element_type_default = None
        mul_tensor: "f16[2048, 768]" = torch.ops.aten.mul.Tensor(gt_scalar, addmm_5);  gt_scalar = addmm_5 = None
        mul_tensor_1: "f16[2048, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        add_tensor: "f16[2048, 768]" = torch.ops.aten.add.Tensor(view_12, mul_tensor_1);  view_12 = mul_tensor_1 = None
        view_default: "f16[4, 512, 768]" = torch.ops.aten.view.default(add_tensor, _shape_param_0);  add_tensor = _shape_param_0 = None
        return view_default

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
