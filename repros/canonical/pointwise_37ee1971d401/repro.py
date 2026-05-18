"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s2_g21
Pattern hash: 37ee1971d401
Shape hash: 934b90da
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, unsqueeze: "i64[1, 1024]", cumsum: "i64[4, 1024]", getitem_3: "f16[4, 1024, 768]", getitem_4: "f16[4, 1024, 768]", getitem_2: "f16[4, 1024, 768]"):
        # No stacktrace found for following nodes
        iota_default: "i64[4]" = torch.ops.prims.iota.default(4, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default: "i64[4, 1]" = torch.ops.aten.unsqueeze.default(iota_default, 1);  iota_default = None
        unsqueeze_default_1: "i64[4, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "i64[4, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        unsqueeze_default_3: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_default_4: "i64[1, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 3)
        unsqueeze_default_5: "i64[1, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        full_default: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        le_tensor: "b8[1, 1, 1024, 1024]" = torch.ops.aten.le.Tensor(unsqueeze_default_5, unsqueeze_default_4)
        bitwise_and_tensor: "b8[1, 1, 1024, 1024]" = torch.ops.aten.bitwise_and.Tensor(full_default, le_tensor);  full_default = le_tensor = None
        index_tensor: "i64[4, 1, 1024, 1]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_default_2, unsqueeze_default_4]);  unsqueeze_default_4 = None
        index_tensor_1: "i64[4, 1, 1, 1024]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_default_2, unsqueeze_default_5]);  cumsum = unsqueeze_default_2 = unsqueeze_default_5 = None
        eq_tensor: "b8[4, 1, 1024, 1024]" = torch.ops.aten.eq.Tensor(index_tensor, index_tensor_1);  index_tensor = index_tensor_1 = None
        bitwise_and_tensor_1: "b8[4, 1, 1024, 1024]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_tensor, eq_tensor);  bitwise_and_tensor = eq_tensor = None
        expand_default: "b8[4, 1, 1024, 1024]" = torch.ops.aten.expand.default(bitwise_and_tensor_1, [4, -1, 1024, 1024]);  bitwise_and_tensor_1 = None
        reshape_default: "f16[4, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_3, [4, 1024, -1, 64]);  getitem_3 = None
        permute_default: "f16[4, 12, 1024, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        reshape_default_1: "f16[4, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_4, [4, 1024, -1, 64]);  getitem_4 = None
        permute_default_1: "f16[4, 12, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None
        reshape_default_2: "f16[4, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_2, [4, 1024, -1, 64]);  getitem_2 = None
        permute_default_2: "f16[4, 12, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1, 3]);  reshape_default_2 = None
        full_default_1: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f16[4, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_default, full_default_2, full_default_1);  expand_default = full_default_2 = full_default_1 = None
        expand_default_1: "f16[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_self, [4, 12, 1024, 1024]);  where_self = None
        return (permute_default, permute_default_1, permute_default_2, expand_default_1)


def _default_make_inputs():
    return [
    torch.randint(0, 2, [1, 1024], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, [4, 1024], dtype=torch.int64, device='cuda'),
    torch.randn(9435648, dtype=torch.float16, device='cuda').as_strided([4, 1024, 768], [2359296, 2304, 1]),  # getitem_3
    torch.randn(9435648, dtype=torch.float16, device='cuda').as_strided([4, 1024, 768], [2359296, 2304, 1]),  # getitem_4
    torch.randn(9435648, dtype=torch.float16, device='cuda').as_strided([4, 1024, 768], [2359296, 2304, 1]),  # getitem_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
