"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s1_g5
Pattern hash: 54f88ef3631e
Shape hash: 42094fa7
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm: "bf16[1024, 1024]", addmm_1: "bf16[1024, 1024]", addmm_2: "bf16[1024, 1024]"):
        # No stacktrace found for following nodes
        iota_default: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[1024]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None
        iota_default_1: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor_1: "i64[1024]" = torch.ops.aten.add.Tensor(iota_default_1, 0);  iota_default_1 = None
        unsqueeze_default: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None
        unsqueeze_default_1: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 1);  unsqueeze_default = None
        unsqueeze_default_2: "i64[1, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        unsqueeze_default_3: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add_tensor_1, 0);  add_tensor_1 = None
        unsqueeze_default_4: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 1);  unsqueeze_default_3 = None
        unsqueeze_default_5: "i64[1, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        le_tensor: "b8[1, 1, 1024, 1024]" = torch.ops.aten.le.Tensor(unsqueeze_default_5, unsqueeze_default_2);  unsqueeze_default_5 = unsqueeze_default_2 = None
        expand_default: "b8[1, 1, 1024, 1024]" = torch.ops.aten.expand.default(le_tensor, [1, -1, 1024, 1024]);  le_tensor = None
        reshape_default: "bf16[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm, [1, 1024, 1024]);  addmm = None
        reshape_default_1: "bf16[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(reshape_default, [1, 1024, -1, 64]);  reshape_default = None
        permute_default: "bf16[1, 16, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None
        reshape_default_2: "bf16[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_1, [1, 1024, 1024]);  addmm_1 = None
        reshape_default_3: "bf16[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_2, [1, 1024, 1024]);  addmm_2 = None
        reshape_default_4: "bf16[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(reshape_default_2, [1, 1024, -1, 64]);  reshape_default_2 = None
        permute_default_1: "bf16[1, 16, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_4, [0, 2, 1, 3]);  reshape_default_4 = None
        reshape_default_5: "bf16[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(reshape_default_3, [1, 1024, -1, 64]);  reshape_default_3 = None
        permute_default_2: "bf16[1, 16, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_5, [0, 2, 1, 3]);  reshape_default_5 = None
        full_default: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "bf16[1, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_default, full_default_1, full_default);  expand_default = full_default_1 = full_default = None
        expand_default_1: "bf16[1, 16, 1024, 1024]" = torch.ops.aten.expand.default(where_self, [1, 16, 1024, 1024]);  where_self = None
        return (permute_default, permute_default_1, permute_default_2, expand_default_1)


def _default_make_inputs():
    return [
    torch.randn([1024, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
