"""
Standalone repro captured via capture_hook.
Label: hf_google/gemma-3-4b-it_infer
Pattern hash: d284769f6e7e
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
    def forward(self, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29):
        # No stacktrace found for following nodes
        full: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
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
        sub: "i64[1, 1, 1000, 1]" = torch.ops.aten.sub.Tensor(unsqueeze_5, 1024)
        gt: "b8[1, 1, 1000, 1000]" = torch.ops.aten.gt.Tensor(unsqueeze_2, sub);  sub = None
        bitwise_and: "b8[1, 1, 1000, 1000]" = torch.ops.aten.bitwise_and.Tensor(full, gt);  full = gt = None
        le: "b8[1, 1, 1000, 1000]" = torch.ops.aten.le.Tensor(unsqueeze_2, unsqueeze_5);  unsqueeze_2 = unsqueeze_5 = None
        bitwise_and_1: "b8[1, 1, 1000, 1000]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, le);  bitwise_and = le = None
        expand: "b8[1, 1, 1000, 1000]" = torch.ops.aten.expand.default(bitwise_and_1, _shape_param_0);  bitwise_and_1 = _shape_param_0 = None
        full_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_2: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_1, full_2);  full_1 = full_2 = None
        expand_1: "bf16[1, 8, 1000, 1000]" = torch.ops.aten.expand.default(where, _shape_param_1);  where = _shape_param_1 = None
        full_3: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_4: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_3, full_4);  full_3 = full_4 = None
        expand_2: "bf16[1, 8, 1000, 1000]" = torch.ops.aten.expand.default(where_1, _shape_param_2);  where_1 = _shape_param_2 = None
        full_5: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_6: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_5, full_6);  full_5 = full_6 = None
        expand_3: "bf16[1, 8, 1000, 1000]" = torch.ops.aten.expand.default(where_2, _shape_param_3);  where_2 = _shape_param_3 = None
        full_7: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_8: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_7, full_8);  full_7 = full_8 = None
        expand_4: "bf16[1, 8, 1000, 1000]" = torch.ops.aten.expand.default(where_3, _shape_param_4);  where_3 = _shape_param_4 = None
        full_9: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_10: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_9, full_10);  full_9 = full_10 = None
        expand_5: "bf16[1, 8, 1000, 1000]" = torch.ops.aten.expand.default(where_4, _shape_param_5);  where_4 = _shape_param_5 = None
        full_11: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_12: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_11, full_12);  full_11 = full_12 = None
        expand_6: "bf16[1, 8, 1000, 1000]" = torch.ops.aten.expand.default(where_5, _shape_param_6);  where_5 = _shape_param_6 = None
        full_13: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_14: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_13, full_14);  full_13 = full_14 = None
        expand_7: "bf16[1, 8, 1000, 1000]" = torch.ops.aten.expand.default(where_6, _shape_param_7);  where_6 = _shape_param_7 = None
        full_15: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_16: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_15, full_16);  full_15 = full_16 = None
        expand_8: "bf16[1, 8, 1000, 1000]" = torch.ops.aten.expand.default(where_7, _shape_param_8);  where_7 = _shape_param_8 = None
        full_17: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_18: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_17, full_18);  full_17 = full_18 = None
        expand_9: "bf16[1, 8, 1000, 1000]" = torch.ops.aten.expand.default(where_8, _shape_param_9);  where_8 = _shape_param_9 = None
        full_19: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_20: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_9: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_19, full_20);  full_19 = full_20 = None
        expand_10: "bf16[1, 8, 1000, 1000]" = torch.ops.aten.expand.default(where_9, _shape_param_10);  where_9 = _shape_param_10 = None
        full_21: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_22: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_10: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_21, full_22);  full_21 = full_22 = None
        expand_11: "bf16[1, 8, 1000, 1000]" = torch.ops.aten.expand.default(where_10, _shape_param_11);  where_10 = _shape_param_11 = None
        full_23: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_24: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_11: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_23, full_24);  full_23 = full_24 = None
        expand_12: "bf16[1, 8, 1000, 1000]" = torch.ops.aten.expand.default(where_11, _shape_param_12);  where_11 = _shape_param_12 = None
        full_25: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_26: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_25, full_26);  full_25 = full_26 = None
        expand_13: "bf16[1, 8, 1000, 1000]" = torch.ops.aten.expand.default(where_12, _shape_param_13);  where_12 = _shape_param_13 = None
        full_27: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_28: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_13: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_27, full_28);  full_27 = full_28 = None
        expand_14: "bf16[1, 8, 1000, 1000]" = torch.ops.aten.expand.default(where_13, _shape_param_14);  where_13 = _shape_param_14 = None
        full_29: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_30: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_14: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_29, full_30);  full_29 = full_30 = None
        expand_15: "bf16[1, 8, 1000, 1000]" = torch.ops.aten.expand.default(where_14, _shape_param_15);  where_14 = _shape_param_15 = None
        full_31: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_32: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_15: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_31, full_32);  full_31 = full_32 = None
        expand_16: "bf16[1, 8, 1000, 1000]" = torch.ops.aten.expand.default(where_15, _shape_param_16);  where_15 = _shape_param_16 = None
        full_33: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_34: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_16: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_33, full_34);  full_33 = full_34 = None
        expand_17: "bf16[1, 8, 1000, 1000]" = torch.ops.aten.expand.default(where_16, _shape_param_17);  where_16 = _shape_param_17 = None
        full_35: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_36: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_17: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_35, full_36);  full_35 = full_36 = None
        expand_18: "bf16[1, 8, 1000, 1000]" = torch.ops.aten.expand.default(where_17, _shape_param_18);  where_17 = _shape_param_18 = None
        full_37: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_38: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_18: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_37, full_38);  full_37 = full_38 = None
        expand_19: "bf16[1, 8, 1000, 1000]" = torch.ops.aten.expand.default(where_18, _shape_param_19);  where_18 = _shape_param_19 = None
        full_39: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_40: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_19: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_39, full_40);  full_39 = full_40 = None
        expand_20: "bf16[1, 8, 1000, 1000]" = torch.ops.aten.expand.default(where_19, _shape_param_20);  where_19 = _shape_param_20 = None
        full_41: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_42: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_20: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_41, full_42);  full_41 = full_42 = None
        expand_21: "bf16[1, 8, 1000, 1000]" = torch.ops.aten.expand.default(where_20, _shape_param_21);  where_20 = _shape_param_21 = None
        full_43: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_44: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_21: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_43, full_44);  full_43 = full_44 = None
        expand_22: "bf16[1, 8, 1000, 1000]" = torch.ops.aten.expand.default(where_21, _shape_param_22);  where_21 = _shape_param_22 = None
        full_45: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_46: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_22: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_45, full_46);  full_45 = full_46 = None
        expand_23: "bf16[1, 8, 1000, 1000]" = torch.ops.aten.expand.default(where_22, _shape_param_23);  where_22 = _shape_param_23 = None
        full_47: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_48: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_23: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_47, full_48);  full_47 = full_48 = None
        expand_24: "bf16[1, 8, 1000, 1000]" = torch.ops.aten.expand.default(where_23, _shape_param_24);  where_23 = _shape_param_24 = None
        full_49: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_50: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_24: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_49, full_50);  full_49 = full_50 = None
        expand_25: "bf16[1, 8, 1000, 1000]" = torch.ops.aten.expand.default(where_24, _shape_param_25);  where_24 = _shape_param_25 = None
        full_51: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_52: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_25: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_51, full_52);  full_51 = full_52 = None
        expand_26: "bf16[1, 8, 1000, 1000]" = torch.ops.aten.expand.default(where_25, _shape_param_26);  where_25 = _shape_param_26 = None
        full_53: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_54: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_26: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_53, full_54);  full_53 = full_54 = None
        expand_27: "bf16[1, 8, 1000, 1000]" = torch.ops.aten.expand.default(where_26, _shape_param_27);  where_26 = _shape_param_27 = None
        full_55: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_56: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_27: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_55, full_56);  full_55 = full_56 = None
        expand_28: "bf16[1, 8, 1000, 1000]" = torch.ops.aten.expand.default(where_27, _shape_param_28);  where_27 = _shape_param_28 = None
        full_57: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_58: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_28: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_57, full_58);  expand = full_57 = full_58 = None
        expand_29: "bf16[1, 8, 1000, 1000]" = torch.ops.aten.expand.default(where_28, _shape_param_29);  where_28 = _shape_param_29 = None
        return (expand_1, expand_2, expand_3, expand_4, expand_5, expand_6, expand_7, expand_8, expand_9, expand_10, expand_11, expand_12, expand_13, expand_14, expand_15, expand_16, expand_17, expand_18, expand_19, expand_20, expand_21, expand_22, expand_23, expand_24, expand_25, expand_26, expand_27, expand_28, expand_29)



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
