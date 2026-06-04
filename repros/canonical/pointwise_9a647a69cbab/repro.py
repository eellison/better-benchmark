"""
Standalone repro captured via capture_hook.
Label: hf_BlenderbotForCausalLM_infer_000
Pattern hash: 9a647a69cbab
Shape hash: d7517139
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(S([32, -1, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]))"

class Repro(torch.nn.Module):
    def forward(self, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24):
        # No stacktrace found for following nodes
        iota_default: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[128]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None
        unsqueeze_default: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None
        unsqueeze_default_1: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 1);  unsqueeze_default = None
        unsqueeze_default_2: "i64[1, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 2);  unsqueeze_default_1 = None
        iota_default_1: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor_1: "i64[128]" = torch.ops.aten.add.Tensor(iota_default_1, 0);  iota_default_1 = None
        unsqueeze_default_3: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add_tensor_1, 0);  add_tensor_1 = None
        unsqueeze_default_4: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 1);  unsqueeze_default_3 = None
        unsqueeze_default_5: "i64[1, 1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        le_tensor: "b8[1, 1, 128, 128]" = torch.ops.aten.le.Tensor(unsqueeze_default_2, unsqueeze_default_5);  unsqueeze_default_2 = unsqueeze_default_5 = None
        expand_default: "b8[32, 1, 128, 128]" = torch.ops.aten.expand.default(le_tensor, _shape_param_0);  le_tensor = _shape_param_0 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full_default, full_default_1);  full_default = full_default_1 = None
        expand_default_1: "f32[32, 32, 128, 128]" = torch.ops.aten.expand.default(where_self, _shape_param_1);  where_self = _shape_param_1 = None
        full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full_default_2, full_default_3);  full_default_2 = full_default_3 = None
        expand_default_2: "f32[32, 32, 128, 128]" = torch.ops.aten.expand.default(where_self_1, _shape_param_2);  where_self_1 = _shape_param_2 = None
        full_default_4: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_5: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_2: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full_default_4, full_default_5);  full_default_4 = full_default_5 = None
        expand_default_3: "f32[32, 32, 128, 128]" = torch.ops.aten.expand.default(where_self_2, _shape_param_3);  where_self_2 = _shape_param_3 = None
        full_default_6: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_7: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_3: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full_default_6, full_default_7);  full_default_6 = full_default_7 = None
        expand_default_4: "f32[32, 32, 128, 128]" = torch.ops.aten.expand.default(where_self_3, _shape_param_4);  where_self_3 = _shape_param_4 = None
        full_default_8: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_9: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_4: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full_default_8, full_default_9);  full_default_8 = full_default_9 = None
        expand_default_5: "f32[32, 32, 128, 128]" = torch.ops.aten.expand.default(where_self_4, _shape_param_5);  where_self_4 = _shape_param_5 = None
        full_default_10: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_11: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_5: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full_default_10, full_default_11);  full_default_10 = full_default_11 = None
        expand_default_6: "f32[32, 32, 128, 128]" = torch.ops.aten.expand.default(where_self_5, _shape_param_6);  where_self_5 = _shape_param_6 = None
        full_default_12: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_13: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_6: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full_default_12, full_default_13);  full_default_12 = full_default_13 = None
        expand_default_7: "f32[32, 32, 128, 128]" = torch.ops.aten.expand.default(where_self_6, _shape_param_7);  where_self_6 = _shape_param_7 = None
        full_default_14: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_15: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_7: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full_default_14, full_default_15);  full_default_14 = full_default_15 = None
        expand_default_8: "f32[32, 32, 128, 128]" = torch.ops.aten.expand.default(where_self_7, _shape_param_8);  where_self_7 = _shape_param_8 = None
        full_default_16: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_17: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_8: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full_default_16, full_default_17);  full_default_16 = full_default_17 = None
        expand_default_9: "f32[32, 32, 128, 128]" = torch.ops.aten.expand.default(where_self_8, _shape_param_9);  where_self_8 = _shape_param_9 = None
        full_default_18: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_19: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_9: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full_default_18, full_default_19);  full_default_18 = full_default_19 = None
        expand_default_10: "f32[32, 32, 128, 128]" = torch.ops.aten.expand.default(where_self_9, _shape_param_10);  where_self_9 = _shape_param_10 = None
        full_default_20: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_21: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_10: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full_default_20, full_default_21);  full_default_20 = full_default_21 = None
        expand_default_11: "f32[32, 32, 128, 128]" = torch.ops.aten.expand.default(where_self_10, _shape_param_11);  where_self_10 = _shape_param_11 = None
        full_default_22: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_23: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_11: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full_default_22, full_default_23);  full_default_22 = full_default_23 = None
        expand_default_12: "f32[32, 32, 128, 128]" = torch.ops.aten.expand.default(where_self_11, _shape_param_12);  where_self_11 = _shape_param_12 = None
        full_default_24: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_25: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_12: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full_default_24, full_default_25);  full_default_24 = full_default_25 = None
        expand_default_13: "f32[32, 32, 128, 128]" = torch.ops.aten.expand.default(where_self_12, _shape_param_13);  where_self_12 = _shape_param_13 = None
        full_default_26: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_27: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_13: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full_default_26, full_default_27);  full_default_26 = full_default_27 = None
        expand_default_14: "f32[32, 32, 128, 128]" = torch.ops.aten.expand.default(where_self_13, _shape_param_14);  where_self_13 = _shape_param_14 = None
        full_default_28: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_29: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_14: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full_default_28, full_default_29);  full_default_28 = full_default_29 = None
        expand_default_15: "f32[32, 32, 128, 128]" = torch.ops.aten.expand.default(where_self_14, _shape_param_15);  where_self_14 = _shape_param_15 = None
        full_default_30: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_31: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_15: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full_default_30, full_default_31);  full_default_30 = full_default_31 = None
        expand_default_16: "f32[32, 32, 128, 128]" = torch.ops.aten.expand.default(where_self_15, _shape_param_16);  where_self_15 = _shape_param_16 = None
        full_default_32: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_33: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_16: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full_default_32, full_default_33);  full_default_32 = full_default_33 = None
        expand_default_17: "f32[32, 32, 128, 128]" = torch.ops.aten.expand.default(where_self_16, _shape_param_17);  where_self_16 = _shape_param_17 = None
        full_default_34: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_35: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_17: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full_default_34, full_default_35);  full_default_34 = full_default_35 = None
        expand_default_18: "f32[32, 32, 128, 128]" = torch.ops.aten.expand.default(where_self_17, _shape_param_18);  where_self_17 = _shape_param_18 = None
        full_default_36: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_37: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_18: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full_default_36, full_default_37);  full_default_36 = full_default_37 = None
        expand_default_19: "f32[32, 32, 128, 128]" = torch.ops.aten.expand.default(where_self_18, _shape_param_19);  where_self_18 = _shape_param_19 = None
        full_default_38: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_39: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_19: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full_default_38, full_default_39);  full_default_38 = full_default_39 = None
        expand_default_20: "f32[32, 32, 128, 128]" = torch.ops.aten.expand.default(where_self_19, _shape_param_20);  where_self_19 = _shape_param_20 = None
        full_default_40: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_41: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_20: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full_default_40, full_default_41);  full_default_40 = full_default_41 = None
        expand_default_21: "f32[32, 32, 128, 128]" = torch.ops.aten.expand.default(where_self_20, _shape_param_21);  where_self_20 = _shape_param_21 = None
        full_default_42: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_43: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_21: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full_default_42, full_default_43);  full_default_42 = full_default_43 = None
        expand_default_22: "f32[32, 32, 128, 128]" = torch.ops.aten.expand.default(where_self_21, _shape_param_22);  where_self_21 = _shape_param_22 = None
        full_default_44: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_45: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_22: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full_default_44, full_default_45);  full_default_44 = full_default_45 = None
        expand_default_23: "f32[32, 32, 128, 128]" = torch.ops.aten.expand.default(where_self_22, _shape_param_23);  where_self_22 = _shape_param_23 = None
        full_default_46: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_47: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_23: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full_default_46, full_default_47);  expand_default = full_default_46 = full_default_47 = None
        expand_default_24: "f32[32, 32, 128, 128]" = torch.ops.aten.expand.default(where_self_23, _shape_param_24);  where_self_23 = _shape_param_24 = None
        return (expand_default_1, expand_default_2, expand_default_3, expand_default_4, expand_default_5, expand_default_6, expand_default_7, expand_default_8, expand_default_9, expand_default_10, expand_default_11, expand_default_12, expand_default_13, expand_default_14, expand_default_15, expand_default_16, expand_default_17, expand_default_18, expand_default_19, expand_default_20, expand_default_21, expand_default_22, expand_default_23, expand_default_24)

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
