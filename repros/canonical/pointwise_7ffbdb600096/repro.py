"""
Standalone repro captured via capture_hook.
Label: hf_GPT2ForSequenceClassification_infer_000
Pattern hash: 7ffbdb600096
Shape hash: 0931dbc0
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
_shapes_config = "(T([8, 1024], i64, gen=Index(8)), S([8, -1, 1024, 1024]), S([8, 12, 1024, 1024]), S([8, 12, 1024, 1024]), S([8, 12, 1024, 1024]), S([8, 12, 1024, 1024]), S([8, 12, 1024, 1024]), S([8, 12, 1024, 1024]), S([8, 12, 1024, 1024]), S([8, 12, 1024, 1024]), S([8, 12, 1024, 1024]), S([8, 12, 1024, 1024]), S([8, 12, 1024, 1024]), S([8, 12, 1024, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, cumsum: "i64[8, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12):
        # No stacktrace found for following nodes
        full_default: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_default: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[1024]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None
        unsqueeze_default: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None
        unsqueeze_default_1: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 1);  unsqueeze_default = None
        unsqueeze_default_2: "i64[1, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 2);  unsqueeze_default_1 = None
        iota_default_1: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor_1: "i64[1024]" = torch.ops.aten.add.Tensor(iota_default_1, 0);  iota_default_1 = None
        unsqueeze_default_3: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add_tensor_1, 0);  add_tensor_1 = None
        unsqueeze_default_4: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 1);  unsqueeze_default_3 = None
        unsqueeze_default_5: "i64[1, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        le_tensor: "b8[1, 1, 1024, 1024]" = torch.ops.aten.le.Tensor(unsqueeze_default_2, unsqueeze_default_5)
        bitwise_and_tensor: "b8[1, 1, 1024, 1024]" = torch.ops.aten.bitwise_and.Tensor(full_default, le_tensor);  full_default = le_tensor = None
        iota_default_2: "i64[8]" = torch.ops.prims.iota.default(8, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_6: "i64[8, 1]" = torch.ops.aten.unsqueeze.default(iota_default_2, 1);  iota_default_2 = None
        unsqueeze_default_7: "i64[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "i64[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        index_tensor: "i64[8, 1, 1024, 1]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_default_8, unsqueeze_default_5]);  unsqueeze_default_5 = None
        index_tensor_1: "i64[8, 1, 1, 1024]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_default_8, unsqueeze_default_2]);  cumsum = unsqueeze_default_8 = unsqueeze_default_2 = None
        eq_tensor: "b8[8, 1, 1024, 1024]" = torch.ops.aten.eq.Tensor(index_tensor, index_tensor_1);  index_tensor = index_tensor_1 = None
        bitwise_and_tensor_1: "b8[8, 1, 1024, 1024]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_tensor, eq_tensor);  bitwise_and_tensor = eq_tensor = None
        expand_default: "b8[8, 1, 1024, 1024]" = torch.ops.aten.expand.default(bitwise_and_tensor_1, _shape_param_0);  bitwise_and_tensor_1 = _shape_param_0 = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_default, full_default_1, full_default_2);  full_default_1 = full_default_2 = None
        expand_default_1: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_self, _shape_param_1);  where_self = _shape_param_1 = None
        full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_4: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_default, full_default_3, full_default_4);  full_default_3 = full_default_4 = None
        expand_default_2: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_self_1, _shape_param_2);  where_self_1 = _shape_param_2 = None
        full_default_5: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_2: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_default, full_default_5, full_default_6);  full_default_5 = full_default_6 = None
        expand_default_3: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_self_2, _shape_param_3);  where_self_2 = _shape_param_3 = None
        full_default_7: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_8: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_3: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_default, full_default_7, full_default_8);  full_default_7 = full_default_8 = None
        expand_default_4: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_self_3, _shape_param_4);  where_self_3 = _shape_param_4 = None
        full_default_9: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_10: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_4: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_default, full_default_9, full_default_10);  full_default_9 = full_default_10 = None
        expand_default_5: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_self_4, _shape_param_5);  where_self_4 = _shape_param_5 = None
        full_default_11: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_12: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_5: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_default, full_default_11, full_default_12);  full_default_11 = full_default_12 = None
        expand_default_6: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_self_5, _shape_param_6);  where_self_5 = _shape_param_6 = None
        full_default_13: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_14: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_6: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_default, full_default_13, full_default_14);  full_default_13 = full_default_14 = None
        expand_default_7: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_self_6, _shape_param_7);  where_self_6 = _shape_param_7 = None
        full_default_15: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_16: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_7: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_default, full_default_15, full_default_16);  full_default_15 = full_default_16 = None
        expand_default_8: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_self_7, _shape_param_8);  where_self_7 = _shape_param_8 = None
        full_default_17: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_18: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_8: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_default, full_default_17, full_default_18);  full_default_17 = full_default_18 = None
        expand_default_9: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_self_8, _shape_param_9);  where_self_8 = _shape_param_9 = None
        full_default_19: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_20: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_9: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_default, full_default_19, full_default_20);  full_default_19 = full_default_20 = None
        expand_default_10: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_self_9, _shape_param_10);  where_self_9 = _shape_param_10 = None
        full_default_21: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_22: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_10: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_default, full_default_21, full_default_22);  full_default_21 = full_default_22 = None
        expand_default_11: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_self_10, _shape_param_11);  where_self_10 = _shape_param_11 = None
        full_default_23: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_24: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_11: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_default, full_default_23, full_default_24);  expand_default = full_default_23 = full_default_24 = None
        expand_default_12: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_self_11, _shape_param_12);  where_self_11 = _shape_param_12 = None
        return (expand_default_1, expand_default_2, expand_default_3, expand_default_4, expand_default_5, expand_default_6, expand_default_7, expand_default_8, expand_default_9, expand_default_10, expand_default_11, expand_default_12)



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
