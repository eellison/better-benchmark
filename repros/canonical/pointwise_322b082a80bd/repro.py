"""
Standalone repro captured via capture_hook.
Label: hf_DistillGPT2_infer_000
Pattern hash: 322b082a80bd
Shape hash: 9c27f029
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
_shapes_config = "(T([32, 512], i64, gen=Index(32)), S([32, -1, 512, 512]), S([32, 12, 512, 512]))"

class Repro(torch.nn.Module):
    def forward(self, cumsum: "i64[32, 512]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        full_default: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_default: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[512]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None
        unsqueeze_default: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None
        unsqueeze_default_1: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 1);  unsqueeze_default = None
        unsqueeze_default_2: "i64[1, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 2);  unsqueeze_default_1 = None
        iota_default_1: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor_1: "i64[512]" = torch.ops.aten.add.Tensor(iota_default_1, 0);  iota_default_1 = None
        unsqueeze_default_3: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_tensor_1, 0);  add_tensor_1 = None
        unsqueeze_default_4: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 1);  unsqueeze_default_3 = None
        unsqueeze_default_5: "i64[1, 1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        le_tensor: "b8[1, 1, 512, 512]" = torch.ops.aten.le.Tensor(unsqueeze_default_2, unsqueeze_default_5)
        bitwise_and_tensor: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_and.Tensor(full_default, le_tensor);  full_default = le_tensor = None
        iota_default_2: "i64[32]" = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_6: "i64[32, 1]" = torch.ops.aten.unsqueeze.default(iota_default_2, 1);  iota_default_2 = None
        unsqueeze_default_7: "i64[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "i64[32, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        index_tensor: "i64[32, 1, 512, 1]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_default_8, unsqueeze_default_5]);  unsqueeze_default_5 = None
        index_tensor_1: "i64[32, 1, 1, 512]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_default_8, unsqueeze_default_2]);  cumsum = unsqueeze_default_8 = unsqueeze_default_2 = None
        eq_tensor: "b8[32, 1, 512, 512]" = torch.ops.aten.eq.Tensor(index_tensor, index_tensor_1);  index_tensor = index_tensor_1 = None
        bitwise_and_tensor_1: "b8[32, 1, 512, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_tensor, eq_tensor);  bitwise_and_tensor = eq_tensor = None
        expand_default: "b8[32, 1, 512, 512]" = torch.ops.aten.expand.default(bitwise_and_tensor_1, _shape_param_0);  bitwise_and_tensor_1 = _shape_param_0 = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[32, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_1, full_default_2);  expand_default = full_default_1 = full_default_2 = None
        expand_default_1: "f32[32, 12, 512, 512]" = torch.ops.aten.expand.default(where_self, _shape_param_1);  where_self = _shape_param_1 = None
        return expand_default_1



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
