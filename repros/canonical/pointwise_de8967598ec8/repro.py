"""
Standalone repro captured via capture_hook.
Label: hf_GPT2ForSequenceClassification_infer
Pattern hash: de8967598ec8
Shape hash: 0105f520
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
    def forward(self, arg0_1: "i64[8, 1024]", _shape_param_0):
        # No stacktrace found for following nodes
        full: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[1024]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None
        unsqueeze: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None
        unsqueeze_1: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None
        iota_1: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_1: "i64[1024]" = torch.ops.aten.add.Tensor(iota_1, 0);  iota_1 = None
        unsqueeze_3: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add_1, 0);  add_1 = None
        unsqueeze_4: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
        unsqueeze_5: "i64[1, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        le: "b8[1, 1, 1024, 1024]" = torch.ops.aten.le.Tensor(unsqueeze_2, unsqueeze_5)
        bitwise_and: "b8[1, 1, 1024, 1024]" = torch.ops.aten.bitwise_and.Tensor(full, le);  full = le = None
        iota_2: "i64[8]" = torch.ops.prims.iota.default(8, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_6: "i64[8, 1]" = torch.ops.aten.unsqueeze.default(iota_2, 1);  iota_2 = None
        unsqueeze_7: "i64[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 2);  unsqueeze_6 = None
        unsqueeze_8: "i64[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
        index: "i64[8, 1, 1024, 1]" = torch.ops.aten.index.Tensor(arg0_1, [unsqueeze_8, unsqueeze_5]);  unsqueeze_5 = None
        index_1: "i64[8, 1, 1, 1024]" = torch.ops.aten.index.Tensor(arg0_1, [unsqueeze_8, unsqueeze_2]);  arg0_1 = unsqueeze_8 = unsqueeze_2 = None
        eq: "b8[8, 1, 1024, 1024]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None
        bitwise_and_1: "b8[8, 1, 1024, 1024]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, eq);  bitwise_and = eq = None
        expand: "b8[8, 1, 1024, 1024]" = torch.ops.aten.expand.default(bitwise_and_1, _shape_param_0);  bitwise_and_1 = _shape_param_0 = None
        full_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_2: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_1, full_2);  full_1 = full_2 = None
        full_3: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_4: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_3, full_4);  full_3 = full_4 = None
        full_5: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_6: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "bf16[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_5, full_6);  full_5 = full_6 = None
        full_7: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_8: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "bf16[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_7, full_8);  full_7 = full_8 = None
        full_9: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_10: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "bf16[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_9, full_10);  full_9 = full_10 = None
        full_11: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_12: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: "bf16[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_11, full_12);  full_11 = full_12 = None
        full_13: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_14: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "bf16[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_13, full_14);  full_13 = full_14 = None
        full_15: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_16: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7: "bf16[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_15, full_16);  full_15 = full_16 = None
        full_17: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_18: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8: "bf16[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_17, full_18);  full_17 = full_18 = None
        full_19: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_20: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_9: "bf16[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_19, full_20);  full_19 = full_20 = None
        full_21: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_22: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_10: "bf16[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_21, full_22);  full_21 = full_22 = None
        full_23: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_24: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_11: "bf16[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_23, full_24);  expand = full_23 = full_24 = None
        return (where, where_1, where_2, where_3, where_4, where_5, where_6, where_7, where_8, where_9, where_10, where_11)



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
