"""
Standalone repro captured via capture_hook.
Label: torchbench_llava_infer_000
Pattern hash: c14f03aac63b
Shape hash: d7517139
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(S([1, -1, 512, 512]))"

class Repro(torch.nn.Module):
    def forward(self, _shape_param_0):
        # No stacktrace found for following nodes
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
        le_tensor: "b8[1, 1, 512, 512]" = torch.ops.aten.le.Tensor(unsqueeze_default_2, unsqueeze_default_5);  unsqueeze_default_2 = unsqueeze_default_5 = None
        expand_default: "b8[1, 1, 512, 512]" = torch.ops.aten.expand.default(le_tensor, _shape_param_0);  le_tensor = _shape_param_0 = None
        full_default: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default, full_default_1);  full_default = full_default_1 = None
        full_default_2: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_2, full_default_3);  full_default_2 = full_default_3 = None
        full_default_4: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_5: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_2: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_4, full_default_5);  full_default_4 = full_default_5 = None
        full_default_6: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_7: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_3: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_6, full_default_7);  full_default_6 = full_default_7 = None
        full_default_8: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_9: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_4: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_8, full_default_9);  full_default_8 = full_default_9 = None
        full_default_10: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_11: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_5: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_10, full_default_11);  full_default_10 = full_default_11 = None
        full_default_12: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_13: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_6: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_12, full_default_13);  full_default_12 = full_default_13 = None
        full_default_14: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_15: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_7: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_14, full_default_15);  full_default_14 = full_default_15 = None
        full_default_16: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_17: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_8: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_16, full_default_17);  full_default_16 = full_default_17 = None
        full_default_18: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_19: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_9: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_18, full_default_19);  full_default_18 = full_default_19 = None
        full_default_20: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_21: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_10: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_20, full_default_21);  full_default_20 = full_default_21 = None
        full_default_22: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_23: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_11: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_22, full_default_23);  full_default_22 = full_default_23 = None
        full_default_24: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_25: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_12: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_24, full_default_25);  full_default_24 = full_default_25 = None
        full_default_26: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_27: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_13: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_26, full_default_27);  full_default_26 = full_default_27 = None
        full_default_28: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_29: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_14: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_28, full_default_29);  full_default_28 = full_default_29 = None
        full_default_30: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_31: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_15: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_30, full_default_31);  full_default_30 = full_default_31 = None
        full_default_32: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_33: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_16: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_32, full_default_33);  full_default_32 = full_default_33 = None
        full_default_34: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_35: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_17: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_34, full_default_35);  full_default_34 = full_default_35 = None
        full_default_36: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_37: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_18: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_36, full_default_37);  full_default_36 = full_default_37 = None
        full_default_38: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_39: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_19: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_38, full_default_39);  full_default_38 = full_default_39 = None
        full_default_40: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_41: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_20: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_40, full_default_41);  full_default_40 = full_default_41 = None
        full_default_42: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_43: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_21: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_42, full_default_43);  full_default_42 = full_default_43 = None
        full_default_44: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_45: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_22: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_44, full_default_45);  full_default_44 = full_default_45 = None
        full_default_46: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_47: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_23: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_46, full_default_47);  full_default_46 = full_default_47 = None
        full_default_48: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_49: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_24: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_48, full_default_49);  full_default_48 = full_default_49 = None
        full_default_50: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_51: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_25: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_50, full_default_51);  full_default_50 = full_default_51 = None
        full_default_52: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_53: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_26: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_52, full_default_53);  full_default_52 = full_default_53 = None
        full_default_54: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_55: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_27: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_54, full_default_55);  full_default_54 = full_default_55 = None
        full_default_56: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_57: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_28: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_56, full_default_57);  full_default_56 = full_default_57 = None
        full_default_58: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_59: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_29: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_58, full_default_59);  full_default_58 = full_default_59 = None
        full_default_60: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_61: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_30: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_60, full_default_61);  full_default_60 = full_default_61 = None
        full_default_62: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_63: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_31: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_62, full_default_63);  expand_default = full_default_62 = full_default_63 = None
        return (where_self, where_self_1, where_self_2, where_self_3, where_self_4, where_self_5, where_self_6, where_self_7, where_self_8, where_self_9, where_self_10, where_self_11, where_self_12, where_self_13, where_self_14, where_self_15, where_self_16, where_self_17, where_self_18, where_self_19, where_self_20, where_self_21, where_self_22, where_self_23, where_self_24, where_self_25, where_self_26, where_self_27, where_self_28, where_self_29, where_self_30, where_self_31)

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
