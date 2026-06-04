"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Reformer_infer_005
Pattern hash: f26b1736413f
Shape hash: d7517139
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(S([12, 64, 64]), S([12, 64, 64]), S([12, 64, 64]))"

class Repro(torch.nn.Module):
    def forward(self, _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[3]" = torch.ops.prims.inductor_seeds.default(3, device(type='cuda', index=0))
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default: "f32[12, 64, 1, 64]" = torch.ops.prims.inductor_random.default([12, 64, 1, 64], inductor_lookup_seed_default, 'randn');  inductor_lookup_seed_default = None
        convert_element_type_default: "f16[12, 64, 1, 64]" = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.float16);  inductor_random_default = None
        unsqueeze_default: "f16[12, 64, 1, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default, 4);  convert_element_type_default = None
        unsqueeze_default_1: "f16[12, 64, 1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 5);  unsqueeze_default = None
        permute_default: "f16[1, 12, 1, 1, 64, 64]" = torch.ops.aten.permute.default(unsqueeze_default_1, [4, 0, 2, 5, 3, 1]);  unsqueeze_default_1 = None
        permute_default_1: "f16[12, 64, 1, 1, 64, 1]" = torch.ops.aten.permute.default(permute_default, [1, 5, 0, 2, 4, 3]);  permute_default = None
        view_default: "f16[12, 64, 64]" = torch.ops.aten.view.default(permute_default_1, _shape_param_0);  permute_default_1 = _shape_param_0 = None
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_1: "f32[12, 64, 1, 64]" = torch.ops.prims.inductor_random.default([12, 64, 1, 64], inductor_lookup_seed_default_1, 'randn');  inductor_lookup_seed_default_1 = None
        convert_element_type_default_1: "f16[12, 64, 1, 64]" = torch.ops.prims.convert_element_type.default(inductor_random_default_1, torch.float16);  inductor_random_default_1 = None
        unsqueeze_default_2: "f16[12, 64, 1, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_1, 4);  convert_element_type_default_1 = None
        unsqueeze_default_3: "f16[12, 64, 1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 5);  unsqueeze_default_2 = None
        permute_default_2: "f16[1, 12, 1, 1, 64, 64]" = torch.ops.aten.permute.default(unsqueeze_default_3, [4, 0, 2, 5, 3, 1]);  unsqueeze_default_3 = None
        permute_default_3: "f16[12, 64, 1, 1, 64, 1]" = torch.ops.aten.permute.default(permute_default_2, [1, 5, 0, 2, 4, 3]);  permute_default_2 = None
        view_default_1: "f16[12, 64, 64]" = torch.ops.aten.view.default(permute_default_3, _shape_param_1);  permute_default_3 = _shape_param_1 = None
        inductor_lookup_seed_default_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2);  inductor_seeds_default = None
        inductor_random_default_2: "f32[12, 64, 1, 64]" = torch.ops.prims.inductor_random.default([12, 64, 1, 64], inductor_lookup_seed_default_2, 'randn');  inductor_lookup_seed_default_2 = None
        convert_element_type_default_2: "f16[12, 64, 1, 64]" = torch.ops.prims.convert_element_type.default(inductor_random_default_2, torch.float16);  inductor_random_default_2 = None
        unsqueeze_default_4: "f16[12, 64, 1, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 4);  convert_element_type_default_2 = None
        unsqueeze_default_5: "f16[12, 64, 1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 5);  unsqueeze_default_4 = None
        permute_default_4: "f16[1, 12, 1, 1, 64, 64]" = torch.ops.aten.permute.default(unsqueeze_default_5, [4, 0, 2, 5, 3, 1]);  unsqueeze_default_5 = None
        permute_default_5: "f16[12, 64, 1, 1, 64, 1]" = torch.ops.aten.permute.default(permute_default_4, [1, 5, 0, 2, 4, 3]);  permute_default_4 = None
        view_default_2: "f16[12, 64, 64]" = torch.ops.aten.view.default(permute_default_5, _shape_param_2);  permute_default_5 = _shape_param_2 = None
        return (view_default, view_default_1, view_default_2)

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
