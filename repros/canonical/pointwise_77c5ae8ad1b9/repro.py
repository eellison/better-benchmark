"""
Standalone repro captured via capture_hook.
Label: hf_GPTJForCausalLM_train_000
Pattern hash: 77c5ae8ad1b9
Shape hash: e460703b
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
_shapes_config = "(T([1, 128], i64, gen=Index(128)), T([1, 128], i64, gen=Index(1)), S([1, -1, 128, 128]), S([1, 16, 128, 128]))"

class Repro(torch.nn.Module):
    def forward(self, unsqueeze: "i64[1, 128]", cumsum: "i64[1, 128]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        iota_default: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_default, 1);  iota_default = None
        unsqueeze_default_1: "i64[1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "i64[1, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        unsqueeze_default_3: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_default_4: "i64[1, 1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 3)
        unsqueeze_default_5: "i64[1, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        full_default: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        le_tensor: "b8[1, 1, 128, 128]" = torch.ops.aten.le.Tensor(unsqueeze_default_5, unsqueeze_default_4)
        bitwise_and_tensor: "b8[1, 1, 128, 128]" = torch.ops.aten.bitwise_and.Tensor(full_default, le_tensor);  full_default = le_tensor = None
        index_tensor: "i64[1, 1, 128, 1]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_default_2, unsqueeze_default_4]);  unsqueeze_default_4 = None
        index_tensor_1: "i64[1, 1, 1, 128]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_default_2, unsqueeze_default_5]);  cumsum = unsqueeze_default_2 = unsqueeze_default_5 = None
        eq_tensor: "b8[1, 1, 128, 128]" = torch.ops.aten.eq.Tensor(index_tensor, index_tensor_1);  index_tensor = index_tensor_1 = None
        bitwise_and_tensor_1: "b8[1, 1, 128, 128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_tensor, eq_tensor);  bitwise_and_tensor = eq_tensor = None
        expand_default: "b8[1, 1, 128, 128]" = torch.ops.aten.expand.default(bitwise_and_tensor_1, _shape_param_0);  bitwise_and_tensor_1 = _shape_param_0 = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[1, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full_default_1, full_default_2);  expand_default = full_default_1 = full_default_2 = None
        expand_default_1: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(where_self, _shape_param_1);  where_self = _shape_param_1 = None
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
