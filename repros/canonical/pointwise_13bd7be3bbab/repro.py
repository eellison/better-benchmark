"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_infer_000
Pattern hash: 13bd7be3bbab
Shape hash: d7517139
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(S([32, -1, 128, 128]), S([32, 6, 128, 128]), S([32, 6, 128, 128]), S([32, 6, 128, 128]), S([32, 6, 128, 128]), S([32, 6, 128, 128]), S([32, 6, 128, 128]), S([32, 6, 128, 128]), S([32, 6, 128, 128]))"

class Repro(torch.nn.Module):
    def forward(self, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # No stacktrace found for following nodes
        full_default: "f32[1, 6, 128, 128]" = torch.ops.aten.full.default([1, 6, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_default: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[128]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None
        unsqueeze_default: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None
        unsqueeze_default_1: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 1);  unsqueeze_default = None
        unsqueeze_default_2: "i64[1, 1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        ge_scalar: "b8[1, 1, 128, 1]" = torch.ops.aten.ge.Scalar(unsqueeze_default_2, 0);  unsqueeze_default_2 = None
        expand_default: "b8[32, 1, 128, 128]" = torch.ops.aten.expand.default(ge_scalar, _shape_param_0);  ge_scalar = _shape_param_0 = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full_default_1, full_default_2);  expand_default = full_default_1 = full_default_2 = None
        add_tensor_1: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(full_default, where_self);  full_default = where_self = None
        expand_default_1: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(add_tensor_1, _shape_param_1);  _shape_param_1 = None
        expand_default_2: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(add_tensor_1, _shape_param_2);  _shape_param_2 = None
        expand_default_3: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(add_tensor_1, _shape_param_3);  _shape_param_3 = None
        expand_default_4: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(add_tensor_1, _shape_param_4);  _shape_param_4 = None
        expand_default_5: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(add_tensor_1, _shape_param_5);  _shape_param_5 = None
        expand_default_6: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(add_tensor_1, _shape_param_6);  _shape_param_6 = None
        expand_default_7: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(add_tensor_1, _shape_param_7);  _shape_param_7 = None
        expand_default_8: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(add_tensor_1, _shape_param_8);  add_tensor_1 = _shape_param_8 = None
        return (expand_default_1, expand_default_2, expand_default_3, expand_default_4, expand_default_5, expand_default_6, expand_default_7, expand_default_8)

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
