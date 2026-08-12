"""
Standalone repro captured via capture_hook.
Label: hf_google/gemma-3-4b-it_infer
Pattern hash: e2bdfd3a6695
Shape hash: d7517139
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        iota: "i64[1000]" = torch.ops.prims.iota.default(1000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[1000]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None
        unsqueeze: "i64[1, 1000]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None
        unsqueeze_1: "i64[1, 1, 1000]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 1, 1000]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None
        iota_1: "i64[1000]" = torch.ops.prims.iota.default(1000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_1: "i64[1000]" = torch.ops.aten.add.Tensor(iota_1, 0);  iota_1 = None
        unsqueeze_3: "i64[1, 1000]" = torch.ops.aten.unsqueeze.default(add_1, 0);  add_1 = None
        unsqueeze_4: "i64[1, 1, 1000]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
        unsqueeze_5: "i64[1, 1, 1000, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        le: "b8[1, 1, 1000, 1000]" = torch.ops.aten.le.Tensor(unsqueeze_2, unsqueeze_5);  unsqueeze_2 = unsqueeze_5 = None
        expand: "b8[1, 1, 1000, 1000]" = torch.ops.aten.expand.default(le, _shape_param_0);  le = _shape_param_0 = None
        full: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_1: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full, full_1);  full = full_1 = None
        expand_1: "bf16[1, 8, 1000, 1000]" = torch.ops.aten.expand.default(where, _shape_param_1);  where = _shape_param_1 = None
        full_2: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_3: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_2, full_3);  full_2 = full_3 = None
        expand_2: "bf16[1, 8, 1000, 1000]" = torch.ops.aten.expand.default(where_1, _shape_param_2);  where_1 = _shape_param_2 = None
        full_4: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_5: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_4, full_5);  full_4 = full_5 = None
        expand_3: "bf16[1, 8, 1000, 1000]" = torch.ops.aten.expand.default(where_2, _shape_param_3);  where_2 = _shape_param_3 = None
        full_6: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_7: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_6, full_7);  full_6 = full_7 = None
        expand_4: "bf16[1, 8, 1000, 1000]" = torch.ops.aten.expand.default(where_3, _shape_param_4);  where_3 = _shape_param_4 = None
        full_8: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_9: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_8, full_9);  expand = full_8 = full_9 = None
        expand_5: "bf16[1, 8, 1000, 1000]" = torch.ops.aten.expand.default(where_4, _shape_param_5);  where_4 = _shape_param_5 = None
        return (expand_1, expand_2, expand_3, expand_4, expand_5)



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
