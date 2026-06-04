"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Longformer_infer_002
Pattern hash: a863783aadbf
Shape hash: 4fa461c7
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([4096, 768], f16), S([4096, 1, 768]), S([4096, 1, 12, 64]), S([12, 4096, 64]), S([192, 768, 64]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_68: "f16[4096, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "f16[4096, 1, 768]" = torch.ops.aten.view.default(addmm_68, _shape_param_0);  addmm_68 = _shape_param_0 = None
        view_default_1: "f16[4096, 1, 12, 64]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        permute_default: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_default_1, [1, 0, 2, 3]);  view_default_1 = None
        permute_default_1: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        view_default_2: "f16[12, 4096, 64]" = torch.ops.aten.view.default(permute_default_1, _shape_param_2);  permute_default_1 = _shape_param_2 = None
        constant_pad_nd_default: "f16[12, 4608, 64]" = torch.ops.aten.constant_pad_nd.default(view_default_2, [0, 0, 256, 256], -1.0);  view_default_2 = None
        as_strided_default: "f16[12, 16, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_default, [12, 16, 768, 64], [294912, 16384, 64, 1]);  constant_pad_nd_default = None
        unsqueeze_default: "f16[12, 16, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_default, 4);  as_strided_default = None
        clone_default: "f16[12, 16, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_default, memory_format = torch.contiguous_format);  unsqueeze_default = None
        view_default_3: "f16[192, 768, 64]" = torch.ops.aten.view.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
        return view_default_3

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
